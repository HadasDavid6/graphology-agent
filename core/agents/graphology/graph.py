"""The graphology agent workflow.

Graph: START -> guard_input -> retrieve -> answer -> END

The corpus is loaded once at import, and the RAG index is built over the
local graphology corpus. All model I/O goes through SafeAIProxy.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.agents.graphology import GRAPH_INDEX
from core.agents.graphology import nodes
from core.runtime.events import Tracer
from core.runtime.orchestrator import traced
from core.runtime.state import AgentState
from core.tools.skills import load_skills

_DIR = Path(__file__).parent
_CFG = yaml.safe_load((_DIR / "agent.yaml").read_text(encoding="utf-8"))
_SKILLS_TEXT = load_skills(_DIR, _CFG.get("skills", []))
_RAG = _CFG.get("rag", {}) or {}
_TOP_K = _RAG.get("top_k", 4)
_EMBED_MODEL = _RAG.get("embed_model", "text-embedding-3-large")
_MIN_SIMILARITY = _RAG.get("min_similarity", 0.12)
_INDEX = GRAPH_INDEX


def _route_after_input(state: AgentState) -> str:
    return "END" if state.get("blocked") else "retrieve"


def _route_after_retrieve(state: AgentState) -> str:
    return "END" if state.get("blocked") else "answer"


class GraphologyAgent:
    name = _CFG["name"]
    config = _CFG

    def __init__(self, mock: bool | None = None, model_override: str | None = None):
        self.mock = mock
        self.model_override = model_override

    def build(self, tracer: Tracer):
        retrieve_node = partial(
            nodes.retrieve,
            index=_INDEX,
            top_k=_TOP_K,
            embed_model=_EMBED_MODEL,
            min_similarity=_MIN_SIMILARITY,
            mock=self.mock,
        )
        answer_node = partial(
            nodes.answer,
            model=self.model_override or _CFG["model"],
            max_output_tokens=_CFG["max_output_tokens"],
            budget_tokens=_CFG["budget_tokens"],
            mock=self.mock,
            skills_text=_SKILLS_TEXT,
        )

        g = StateGraph(AgentState)
        g.add_node("guard_input", traced("guard_input", nodes.guard_input, tracer))
        g.add_node("retrieve", traced("retrieve", retrieve_node, tracer))
        g.add_node("answer", traced("answer", answer_node, tracer))

        g.add_edge(START, "guard_input")
        g.add_conditional_edges("guard_input", _route_after_input,
                                {"retrieve": "retrieve", "END": END})
        g.add_conditional_edges("retrieve", _route_after_retrieve,
                                {"answer": "answer", "END": END})
        g.add_edge("answer", END)

        return g.compile(checkpointer=MemorySaver())


def make_agent(mock: bool | None = None, model_override: str | None = None) -> GraphologyAgent:
    return GraphologyAgent(mock=mock, model_override=model_override)
