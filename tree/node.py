from dataclasses import dataclass, field
from typing import Literal, Optional, List

@dataclass
class NodeState:
    node_id: str
    parent_id: Optional[str]
    node_type: Literal["technique", "implementation"]
    plan: Optional[str] = None          # Technique node output: strategy/plan
    code: Optional[str] = None          # Implementation node output: glue code
    config: Optional[dict] = None       # Config details
    result: Optional[dict] = None       # {"score": float, "diagnostics": str}
    executed: bool = False               # True after node has been fully processed
    
    # Scheduling fields
    visits: int = 0
    total_reward: float = 0.0
    children_ids: List[str] = field(default_factory=list)

