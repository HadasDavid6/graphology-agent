
"""Node functions for the graphology agent.

Flow: guard_input -> retrieve -> answer.
All model I/O goes through SafeAIProxy, and queries are grounded in the local
graphology corpus.
"""

from __future__ import annotations

from time import perf_counter

from core.guardrails import policies
from core.guardrails.safeai_proxy import SafeAIProxy
from core.memory.short_term import build_messages
from core.providers.openrouter import BudgetExceeded, OpenRouterProvider
from core.runtime.state import AgentState
from core.tools.rag import EmbeddingIndex, tokenize


SYSTEM_PROMPT = (
    "אתה עוזר המסביר את המסורת והגישות של גרפולוגיה, ולא מומחה שמאבחן אנשים. "
    "ענה בעברית תקנית וברורה, ורק על בסיס המדריך המקומי. הצג פרשנויות גרפולוגיות "
    "כטענות של התחום ולא כעובדות מוכחות: גרפולוגיה אינה כלי מדעי מהימן להערכת אישיות. "
    "הבהר שניתוח כתב-יד אינו אבחנה רפואית או פסיכולוגית, ואל תיתן עצות טיפוליות. "
    "אם חסר מידע, אמור זאת ובקש תיאור של צורות האותיות, השיפוע, הרווחים או לחץ העט. "
    "אל תסיק מסקנות ודאיות על אדם מסימן יחיד."
)

_MOCK_STOP_WORDS = {
    "איך", "האם", "הוא", "היא", "זה", "מה", "של", "על", "את", "עם", "לפי",
    "אני", "יש", "בין", "רק", "אפשר", "משמעות", "אומר", "אומרים",
}

_MOCK_DOMAIN_TERMS = {
    "גרפולוג", "כתב", "כתיבה", "אות", "שיפוע", "נטייה", "רווח", "לחץ",
    "חתימה", "שורה", "אישיות",
}

_DOMAIN_QUERY_TERMS = {
    "גרפולוג", "כתב", "כתיב", "אות", "שיפוע", "נטייה", "נטיה", "רווח",
    "לחץ", "חתימ", "שורה", "אישיות",
}


def _system_prompt(skills_text: str) -> str:
    if skills_text:
        return SYSTEM_PROMPT + "\n\n" + skills_text
    return SYSTEM_PROMPT


def _is_graphology_query(query: str) -> bool:
    terms = tokenize(query)
    return any(
        term.startswith(domain_term) or domain_term.startswith(term)
        for term in terms
        for domain_term in _DOMAIN_QUERY_TERMS
    )


def _number(value) -> float:
    """Convert a usage value to a safe numeric value."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _add_usage(state: AgentState, **values) -> dict:
    """Return a new usage dictionary, preserving metrics from earlier nodes."""
    usage = dict(state.get("usage") or {})

    for key, value in values.items():
        usage[key] = _number(usage.get(key)) + _number(value)

    usage["total_cost_usd"] = (
        _number(usage.get("cost_usd"))
        + _number(usage.get("embedding_cost_usd"))
    )
    usage["total_tokens"] = (
        _number(usage.get("input_tokens"))
        + _number(usage.get("output_tokens"))
        + _number(usage.get("embedding_tokens"))
    )

    return usage


def _mock_search(query: str, index: EmbeddingIndex, top_k: int):
    """Use local token overlap when embeddings are unavailable in mock mode."""
    query_terms = {
        term for term in tokenize(query)
        if len(term) > 2 and term not in _MOCK_STOP_WORDS
    }

    if not query_terms or not any(
        domain_term in term or term in domain_term
        for term in query_terms
        for domain_term in _MOCK_DOMAIN_TERMS
    ):
        return []

    scored = []
    for chunk in index.chunks:
        chunk_terms = set(chunk.tokens or tokenize(chunk.text))
        matches = {
            term for term in query_terms
            if any(term in candidate or candidate in term for candidate in chunk_terms)
        }
        if matches:
            scored.append((len(matches), chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(chunk, 1.0) for _matches, chunk in scored[:top_k]]


def guard_input(state: AgentState) -> dict:
    """Input guardrail as an explicit, traceable step."""
    res = policies.check_input(state.get("user_input", ""))

    if not res.ok:
        return {
            "blocked": True,
            "block_reason": res.reason,
            "final_answer": "לא ניתן לעבד את הבקשה (הקלט לא תקין). נסי לנסח מחדש.",
        }

    return {"blocked": False}


def retrieve(
    state: AgentState,
    *,
    index: EmbeddingIndex,
    top_k: int,
    embed_model: str,
    min_similarity: float = 0.12,
    mock: bool | None = None,
) -> dict:
    """Retrieve relevant chunks and record embedding usage and duration."""
    if not _is_graphology_query(state["user_input"]):
        return {
            "blocked": True,
            "block_reason": "Question is outside the graphology domain.",
            "final_answer": (
                "השאלה חורגת מתחום הגרפולוגיה, ולכן אני לא יכולה לענות עליה. "
                "אני יכולה לעזור בשאלות על פרשנויות גרפולוגיות של כתב-יד."
            ),
            "retrieved": "",
            "sources": [],
        }

    started = perf_counter()

    provider = OpenRouterProvider(
        model=embed_model,
        max_output_tokens=0,
        budget_tokens=None,
        mock=mock,
    )

    hits = index.search(state["user_input"], provider=provider, k=top_k)

    # Capture usage immediately after embedding, before another provider call.
    embedding_usage = dict(getattr(provider, "last_usage", {}) or {})

    if provider.mock and not hits:
        hits = _mock_search(state["user_input"], index, top_k)

    elapsed = perf_counter() - started

    usage = _add_usage(
        state,
        embedding_tokens=embedding_usage.get("input_tokens", 0),
        embedding_cost_usd=embedding_usage.get("cost_usd", 0),
        retrieval_seconds=elapsed,
    )

    if not hits or hits[0][1] < min_similarity:
        return {
            "blocked": True,
            "block_reason": "No sufficiently relevant information in corpus.",
            "final_answer": (
                "לא מצאתי מידע מתאים לשאלה במדריך הגרפולוגיה המקומי, "
                "ולכן אני לא יכולה לענות עליה על סמך המקורות. "
                "אפשר לשאול על שיפוע הכתב, רווחים, לחץ העט או צורות האותיות."
            ),
            "retrieved": "",
            "sources": [],
            "usage": usage,
        }

    blocks, sources = [], []

    for chunk, _score in hits:
        sources.append(chunk.doc)
        blocks.append(f"[מתוך: {chunk.doc}]\n{chunk.text}")

    return {
        "retrieved": "\n\n---\n\n".join(blocks),
        "sources": sorted(set(sources)),
        "usage": usage,
    }


def answer(
    state: AgentState,
    *,
    model: str,
    max_output_tokens: int,
    budget_tokens: int,
    mock: bool | None,
    skills_text: str = "",
) -> dict:
    """Generate an answer and record completion usage and duration."""
    if state.get("blocked"):
        return {"final_answer": state.get("final_answer", "")}

    provider = OpenRouterProvider(
        model=model,
        max_output_tokens=max_output_tokens,
        budget_tokens=budget_tokens,
        mock=mock,
    )
    proxy = SafeAIProxy(provider)

    ctx = state.get("retrieved", "")
    user = state["user_input"]

    if ctx:
        system = (
            _system_prompt(skills_text)
            + "\n\n"
            + "הנה קטעים מהמדריך (השתמש רק במידע מהם):\n"
            + ctx
        )
    else:
        system = (
            _system_prompt(skills_text)
            + "\n\n"
            + "(לא נמצאו קטעים רלוונטיים במדריך הגרפולוגיה.)"
        )

    messages = build_messages(system, user, state.get("history"))

    existing_usage = state.get("usage") or {}
    used = (
        _number(existing_usage.get("input_tokens"))
        + _number(existing_usage.get("output_tokens"))
    )

    started = perf_counter()

    try:
        result = proxy.complete(messages, used_tokens=used)
    except BudgetExceeded as exc:
        elapsed = perf_counter() - started
        usage = _add_usage(
            state,
            generation_seconds=elapsed,
        )
        return {
            "blocked": True,
            "block_reason": str(exc),
            "final_answer": "הבקשה חרגה מתקציב הטוקנים שהוגדר לאג'נט.",
            "usage": usage,
        }

    elapsed = perf_counter() - started
    completion = result.completion

    usage = _add_usage(
        state,
        input_tokens=completion.input_tokens if completion else 0,
        output_tokens=completion.output_tokens if completion else 0,
        cost_usd=completion.cost_usd if completion else 0,
        generation_seconds=elapsed,
    )

    if result.blocked:
        return {
            "blocked": True,
            "block_reason": result.reason,
            "final_answer": "התשובה נחסמה על ידי שכבת ה-guardrails.",
            "usage": usage,
        }

    return {
        "draft_answer": result.text,
        "final_answer": result.text,
        "usage": usage,
        "sources": state.get("sources", []),
    }
