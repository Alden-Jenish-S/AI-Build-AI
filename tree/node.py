"""In-memory state for one planning or implementation branch."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeState:
    node_id: str
    parent_id: str | None
    node_type: str = "implementation"
    plan: str | None = None
    code: str | None = None
    result: dict[str, Any] | None = None
    executed: bool = False
    operator: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    children_ids: list[str] = field(default_factory=list)
