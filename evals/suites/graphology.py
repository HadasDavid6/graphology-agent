"""Checks for the graphology eval suite.

Each check is a function(final_state, expected) -> (passed, detail). The runner maps
the dataset's `checks` keys to these functions.
"""

from __future__ import annotations

# Keywords that indicate the answer stays inside the graphology domain.
_GRAPHOLOGY_HINTS = [
    "גרפולוג",
    "גרפולוגיה",
    "כתב",
    "כתב-יד",
    "אותיות",
    "שיפוע",
    "רווח",
    "לחץ",
    "חתימה",
    "קו בסיס",
]

# Keywords that indicate the answer includes an uncertainty / non-diagnostic caveat.
_LIMITATION_HINTS = [
    "אבחון",
    "אבחנה",
    "לא ניתן לקבוע",
    "לא מספיק מידע",
    "אינו כלי מהימן",
    "אין מחקר",
    "פרטים נוספים",
    "דוגמת כתב",
]

def check_not_blocked(final, expected) -> tuple[bool, str]:
    blocked = bool(final.get("blocked"))
    want_not_blocked = bool(expected)
    passed = (not blocked) == want_not_blocked
    return passed, f"blocked={blocked} (expected not_blocked={want_not_blocked})"


def check_non_empty(final, expected) -> tuple[bool, str]:
    text = (final.get("final_answer") or "").strip()
    return (len(text) > 0) == bool(expected), f"len={len(text)}"


def check_contains_any(final, expected) -> tuple[bool, str]:
    text = final.get("final_answer") or ""
    hits = [w for w in expected if w in text]
    return (len(hits) > 0), f"hits={hits}"


def check_mentions_graphology(final, expected) -> tuple[bool, str]:
    text = final.get("final_answer") or ""
    hits = [w for w in _GRAPHOLOGY_HINTS if w in text]
    passed = (len(hits) > 0) == bool(expected)
    return passed, f"graphology_hints={hits}"


def check_mentions_limitations(final, expected) -> tuple[bool, str]:
    text = final.get("final_answer") or ""
    hits = [w for w in _LIMITATION_HINTS if w in text]
    passed = (len(hits) > 0) == bool(expected)
    return passed, f"limitation_hints={hits}"


def check_has_sources(final, expected) -> tuple[bool, str]:
    sources = final.get("sources") or []
    passed = (len(sources) > 0) == bool(expected)
    return passed, f"sources={sources}"


def check_sources_include_any(final, expected) -> tuple[bool, str]:
    sources = final.get("sources") or []
    hits = [source for source in sources if source in expected]
    return len(hits) > 0, f"sources={sources}, expected_any={expected}, hits={hits}"


def check_retrieved_non_empty(final, expected) -> tuple[bool, str]:
    retrieved = (final.get("retrieved") or "").strip()
    return (len(retrieved) > 0) == bool(expected), f"len={len(retrieved)}"


def check_block_reason_contains_any(final, expected) -> tuple[bool, str]:
    reason = final.get("block_reason") or ""
    hits = [w for w in expected if w in reason]
    return len(hits) > 0, f"block_reason={reason!r}, hits={hits}"


def _trace_nodes(final) -> list[str]:
    return [
        ev.get("node", "")
        for ev in final.get("trace", [])
        if ev.get("type") == "node_start"
    ]


def check_trace_contains(final, expected) -> tuple[bool, str]:
    nodes = _trace_nodes(final)
    missing = [node for node in expected if node not in nodes]
    return len(missing) == 0, f"nodes={nodes}, missing={missing}"


def check_trace_not_contains(final, expected) -> tuple[bool, str]:
    nodes = _trace_nodes(final)
    hits = [node for node in expected if node in nodes]
    return len(hits) == 0, f"nodes={nodes}, forbidden_hits={hits}"


CHECKS = {
    "not_blocked": check_not_blocked,
    "non_empty": check_non_empty,
    "contains_any": check_contains_any,
    "mentions_graphology": check_mentions_graphology,
    "mentions_limitations": check_mentions_limitations,
    "has_sources": check_has_sources,
    "sources_include_any": check_sources_include_any,
    "retrieved_non_empty": check_retrieved_non_empty,
    "block_reason_contains_any": check_block_reason_contains_any,
    "trace_contains": check_trace_contains,
    "trace_not_contains": check_trace_not_contains,
}
