"""Small lineage-aware UCB scheduler for the pending planning frontier."""

from __future__ import annotations

import math

from .node import NodeState


class UCB1Scheduler:
    """Favor strong lineages while retaining bounded exploration."""

    def __init__(self, total_budget: int, exploration: float = 1.15) -> None:
        self.total_budget = max(1, int(total_budget))
        self.exploration = max(0.0, float(exploration))
        self.current_step = 0

    @staticmethod
    def _root_branch(
        node_id: str,
        root_id: str,
        all_nodes: dict[str, NodeState],
    ) -> NodeState:
        current = all_nodes[node_id]
        while current.parent_id not in {None, root_id}:
            parent = all_nodes.get(str(current.parent_id))
            if parent is None:
                break
            current = parent
        return current

    def backpropagate(
        self,
        node_id: str,
        reward: float,
        all_nodes: dict[str, NodeState],
    ) -> None:
        """Propagate one measured reward through its structural lineage."""
        current_id: str | None = node_id
        visited: set[str] = set()
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            current = all_nodes.get(current_id)
            if current is None:
                break
            current.visits += 1
            current.total_reward += float(reward)
            current_id = current.parent_id

    def frontier_scores(
        self,
        root_id: str,
        all_nodes: dict[str, NodeState],
    ) -> dict[str, float]:
        """Return pending reachable nodes and their lineage-level UCB scores."""
        root = all_nodes.get(root_id)
        if root is None:
            return {}

        pending: list[NodeState] = []
        queue = [root_id]
        seen: set[str] = set()
        while queue:
            node_id = queue.pop(0)
            if node_id in seen:
                continue
            seen.add(node_id)
            node = all_nodes.get(node_id)
            if node is None:
                continue
            if node_id != root_id and not node.executed:
                pending.append(node)
                continue
            queue.extend(child for child in node.children_ids if child in all_nodes)

        scores: dict[str, float] = {}
        root_visits = max(root.visits, 1)
        for candidate in pending:
            branch = self._root_branch(candidate.node_id, root_id, all_nodes)
            branch_mean = (
                branch.total_reward / branch.visits if branch.visits else 0.0
            )
            exploration = self.exploration * math.sqrt(
                math.log(root_visits + 2.0) / (branch.visits + 1.0)
            )
            priority = float((candidate.config or {}).get("priority", 0.0) or 0.0)
            scores[candidate.node_id] = branch_mean + exploration + priority
        return scores

    def select_next_node(
        self,
        root_id: str,
        all_nodes: dict[str, NodeState],
        eligible_node_ids: set[str] | None = None,
    ) -> str | None:
        scores = self.frontier_scores(root_id, all_nodes)
        if eligible_node_ids is not None:
            scores = {
                node_id: score
                for node_id, score in scores.items()
                if node_id in eligible_node_ids
            }
        if not scores:
            return None
        return max(scores, key=lambda node_id: (scores[node_id], -_node_number(node_id)))


def _node_number(node_id: str) -> int:
    digits = "".join(character for character in str(node_id) if character.isdigit())
    return int(digits) if digits else 0
