"""Small lineage-aware UCB scheduler for the pending planning frontier."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from search_evidence import pearson_correlation

from .node import NodeState

#: How much a frontier action is boosted when its measured parent's
#: predictions are decorrelated with the incumbent best. 0.0 disables the term.
_COMPLEMENTARITY_WEIGHT = float(0.4)


class UCB1Scheduler:
    """Favor strong lineages while retaining bounded exploration.

    Pending actions under the same measured parent are alternatives with the
    same prior, so within one lineage selection reduces to operator priority;
    across lineages the value term uses the best reward measured anywhere in
    that lineage rather than a mean diluted by weak roots.
    """

    def __init__(self, total_budget: int, exploration: float = 1.15) -> None:
        self.total_budget = max(1, int(total_budget))
        self.exploration = max(0.0, float(exploration))



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
            current.best_reward = max(current.best_reward, float(reward))
            current_id = current.parent_id

    def frontier_scores(
        self,
        root_id: str,
        all_nodes: dict[str, NodeState],
        *,
        best_signature: list[float] | None = None,
        signature_provider: Callable[[NodeState], list[float] | None] | None = None,
        complementarity_weight: float = _COMPLEMENTARITY_WEIGHT,
    ) -> dict[str, float]:
        """Return pending reachable nodes and their lineage-level UCB scores.

        ``best_signature`` / ``signature_provider`` optionally add a
        prediction-complementarity bonus: pending actions whose measured
        parent is decorrelated with the incumbent best are favored, which
        steers exploration toward genuinely new signal rather than more of
        the same representation.
        """
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

        use_complementarity = (
            complementarity_weight > 0.0
            and best_signature is not None
            and signature_provider is not None
        )
        scores: dict[str, float] = {}
        root_visits = max(root.visits, 1)
        for candidate in pending:
            branch = all_nodes.get(str(candidate.parent_id))
            if branch is None:
                branch = root
            branch_best = branch.best_reward
            branch_value = branch_best if math.isfinite(branch_best) else 0.0
            exploration = self.exploration * math.sqrt(
                math.log(root_visits + 2.0) / (branch.visits + 1.0)
            )
            priority = float((candidate.config or {}).get("priority", 0.0) or 0.0)
            score = branch_value + exploration + priority
            if use_complementarity:
                signature = signature_provider(candidate)
                if signature is not None:
                    correlation = pearson_correlation(best_signature, signature)
                    score += complementarity_weight * (1.0 - abs(correlation))
            scores[candidate.node_id] = score
        return scores

    def select_next_node(
        self,
        root_id: str,
        all_nodes: dict[str, NodeState],
        eligible_node_ids: set[str] | None = None,
        **selection_kwargs: Any,
    ) -> str | None:
        scores = self.frontier_scores(root_id, all_nodes, **selection_kwargs)
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
