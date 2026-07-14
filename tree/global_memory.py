from typing import Dict, Any, List, Optional
from .node import NodeState

class GlobalMemory:
    def __init__(self):
        # Maps node_id -> dict representing the record (type, delta/state, score, diagnostics)
        self.records: Dict[str, dict] = {}
        # Log of queries made outside default context for analytics
        self.query_log: List[dict] = []

    def record_technique(self, node_id: str, delta_technique_pool: str, diagnostics: str):
        """Records metadata for a technique node."""
        self.records[node_id] = {
            "type": "technique",
            "delta_technique_pool": delta_technique_pool,
            "diagnostics": diagnostics
        }

    def record_implementation(self, node_id: str, node_state: dict, score: float, diagnostics: str):
        """Records execution metadata and score for an implementation node."""
        self.records[node_id] = {
            "type": "implementation",
            "node_state": node_state,
            "score": score,
            "diagnostics": diagnostics
        }

    def get_default_context(self, node_id: str, all_nodes: Dict[str, NodeState]) -> Dict[str, Any]:
        """
        Computes the default context for node v:
        default_context(v) = {record[parent(v)]} U {record[u] for u in siblings(v)}
        """
        context = {}
        node = all_nodes.get(node_id)
        if not node:
            return context
            
        parent_id = node.parent_id
        if parent_id and parent_id in self.records:
            context["parent"] = self.records[parent_id]
            
            # Sibling nodes share the same parent_id
            parent_node = all_nodes[parent_id]
            siblings = []
            for cid in parent_node.children_ids:
                if cid != node_id and cid in self.records:
                    siblings.append(self.records[cid])
            context["siblings"] = siblings
            
        return context

    def query_record(self, caller_node_id: str, target_node_id: str) -> Optional[dict]:
        """Queries a record directly by node_id (outside of default context).
        Logs this query for metric logging.
        """
        # Log the cross-context query event
        self.query_log.append({
            "caller": caller_node_id,
            "target": target_node_id,
            "success": target_node_id in self.records
        })
        return self.records.get(target_node_id)
