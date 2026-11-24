"""OpenRouter provider.

Owns the single client that talks to the model. OpenRouter only gives us completions +
tool calling; everything else (RAG, browser, code-exec) is implemented elsewhere as tools.

Two modes:
  * real  -> POST to OpenRouter (OpenAI-compatible chat/completions). Requires
             OPENROUTER_API_KEY.
  * mock  -> returns a canned response without any network call. Lets you run and inspect
             the full pipeline before wiring a key. Auto-selected when the key is missing,
             or forced with mock=True.

Budget caps are enforced here: a run that exceeds its token budget raises BudgetExceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import httpx

from core.security.api_keys import resolve_openrouter_key

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"


class BudgetExceeded(Exception):
    def __init__(self, message: str, usage: dict[str, int | float | None]):
        super().__init__(message)
        self.usage = usage


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None


class OpenRouterProvider:
    def __init__(self, *, model: str, max_output_tokens: int = 1024,
                 budget_tokens: int | None = None, mock: bool | None = None):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.budget_tokens = budget_tokens
        self._api_key = resolve_openrouter_key()
        # mock if explicitly requested, or if no key is available
        self.mock = mock if mock is not None else (self._api_key is None)
        self.last_usage: dict[str, int | float | None] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": None,
        }

    def complete(self, messages: list[dict], *, used_tokens: int = 0) -> Completion:
        if self.mock:
            completion = self._mock_complete(messages)
        else:
            completion = self._real_complete(messages, used_tokens=used_tokens)
        self.last_usage = {
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "cost_usd": completion.cost_usd,
        }
        return completion

    # --- mock -------------------------------------------------------------
    def _mock_complete(self, messages: list[dict]) -> Completion:
        user_turns = [m for m in messages if m["role"] == "user"]
        last_user = user_turns[-1]["content"] if user_turns else ""
        if len(user_turns) <= 1:
            # first turn: simulate the skill's "ask clarifying questions" behavior
            text = (
                "‎[MOCK] כדי לאבחן מדויק, כמה שאלות:\n"
                "1. היכן בדיוק רואים מים — על הברז, בחיבור הגמיש, או בסיפון?\n"
                "2. סוג הברז (פרח / שתי ידיות / קיר)?\n"
                "3. שנת בניית הבית בקירוב?\n"
                "(זו תשובת mock — הגדירי OPENROUTER_API_KEY לתשובות אמיתיות.)"
            )
        else:
            text = (
                "‎[MOCK] תודה על הפרטים. בהתבסס על מה שתיארת, הצעד הבא הוא לסגור את "
                "ברזי הניתוק, לייבש, ולאתר את נקודת הנזילה עם נייר יבש. "
                f"(התייחסות אחרונה: «{last_user[:80]}».)"
            )
        return Completion(text=text, input_tokens=len(last_user) // 4 + 20,
                          output_tokens=len(text) // 4, cost_usd=0.0)
    # --- real -------------------------------------------------------------
    def _real_complete(self, messages: list[dict], *, used_tokens: int) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        with httpx.Client(timeout=60) as client:
            resp = client.post(OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = usage.get("cost", data.get("cost"))
        if cost is not None and (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError("OpenRouter returned an invalid usage cost")

        if self.budget_tokens is not None and used_tokens + in_tok + out_tok > self.budget_tokens:
            raise BudgetExceeded(
                f"token budget {self.budget_tokens} exceeded "
                f"({used_tokens + in_tok + out_tok})",
                usage={
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": float(cost) if cost is not None else None,
                },
            )

        return Completion(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=float(cost) if cost is not None else None,
        )

    def embed(self, inputs: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
        if self.mock:
            input_tokens = sum(len(text) // 4 for text in inputs)
            self.last_usage = {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
            return [[0.0] * 1536 for _ in inputs]

        payload = {
            "model": model,
            "input": inputs,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60) as client:
            resp = client.post(EMBEDDING_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        usage = data.get("usage", {}) or {}
        cost = usage.get("cost", data.get("cost"))
        if cost is not None and (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError("OpenRouter returned an invalid embedding usage cost")
        self.last_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": 0,
            "cost_usd": float(cost) if cost is not None else None,
        }
        return [item["embedding"] for item in data["data"]]
