"""Agent registry.

Maps agent names to their factory. Adding an agent = add its folder (graph.py +
agent.yaml) and register its factory here. The runtime/CLI/server discover agents
only through this registry.
"""

from __future__ import annotations

from typing import Callable

from core.agents.device_guide.graph import make_agent as make_device_guide
from core.agents.graphology.graph import make_agent as make_graphology
from core.agents.plumber.graph import make_agent as make_plumber

# name -> factory(mock: bool | None, model_override: str | None) -> agent
_REGISTRY: dict[str, Callable] = {
    "graphology": make_graphology,
    "plumber": make_plumber,
    "device_guide": make_device_guide,
}


def list_agents() -> list[str]:
    return sorted(_REGISTRY)


def get_agent(name: str, mock: bool | None = None, model_override: str | None = None):
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent '{name}'. available: {', '.join(list_agents())}")
    return _REGISTRY[name](mock=mock, model_override=model_override)
