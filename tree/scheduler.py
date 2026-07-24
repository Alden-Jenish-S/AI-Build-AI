import math
from typing import Dict, List, Optional
from .node import NodeState

class UCB1Scheduler:
    def __init__(self, total_budget: int, c_0: float = 1.414, **_legacy_options):
        """Keep scheduling deliberately small.

        Statistical pruning, promotion, diversity, and information gain are
        computed before an action reaches this class. Legacy decay options are
        accepted for configuration compatibility but no longer alter scheduling.
        """
        self.total_budget = total_budget
        self.c_0 = c_0
        self.current_step = 0
        self.warmup_budget = 0

    def set_warmup_budget(self, warmup_budget: int) -> None:
        """Exclude forced initial coverage from UCB decay and visit accounting."""
        self.warmup_budget = max(0, min(int(warmup_budget), self.total_budget))

    def get_exploration_constant(self, t: int) -> float:
        """Return the standard UCB1 exploration constant.

        The previous hand-tuned time decay added scheduler state without
        evidence that it improved time-to-best. Action policies now provide the
        evidence-derived utility through ``config["priority"]``.
        """
        return self.c_0

    def compute_ucb_score(self, node: NodeState, parent_visits: int, t: int) -> float:
        """Calculates the UCB1 score for a node. Unvisited nodes receive infinity."""
        if node.visits == 0:
            return float('inf')
        
        avg_reward = node.total_reward / node.visits
        c_t = self.get_exploration_constant(t)
        
        # Avoid math log(0)
        p_visits = max(parent_visits, 1)
        exploration_term = c_t * math.sqrt(math.log(p_visits) / node.visits)
        
        return avg_reward + exploration_term

    def backpropagate(self, node_id: str, reward: float, all_nodes: Dict[str, NodeState]):
        """Backpropagates the reward up the parent hierarchy path to the root."""
        curr_id = node_id
        while curr_id is not None:
            curr_node = all_nodes[curr_id]
            curr_node.visits += 1
            curr_node.total_reward += reward
            curr_id = curr_node.parent_id

    @staticmethod
    def _root_branch(node_id: str, root_id: str, all_nodes: Dict[str, NodeState]) -> NodeState:
        current = all_nodes[node_id]
        while current.parent_id not in {None, root_id}:
            current = all_nodes[current.parent_id]
        return current

    def compute_frontier_score(
        self,
        node: NodeState,
        root_id: str,
        all_nodes: Dict[str, NodeState],
    ) -> float:
        """Apply standard lineage-level UCB to an already eligible action."""
        root = all_nodes[root_id]
        branch = self._root_branch(node.node_id, root_id, all_nodes)
        branch_mean = branch.total_reward / branch.visits if branch.visits else 0.0
        eligible_root_visits = max(root.visits - self.warmup_budget, 0)
        eligible_branch_visits = max(branch.visits - (1 if branch.visits else 0), 0)
        exploration = self.get_exploration_constant(self.current_step) * math.sqrt(
            math.log(eligible_root_visits + 2.0) / (eligible_branch_visits + 1.0)
        )
        config = node.config or {}
        prior = float(config.get("priority", 0.0) or 0.0)
        return branch_mean + exploration + prior

    def frontier_scores(
        self, root_id: str, all_nodes: Dict[str, NodeState]
    ) -> Dict[str, float]:
        """Return the currently reachable frontier and its auditable scores."""
        if root_id not in all_nodes:
            return {}

        queue = [root_id]
        pending: List[NodeState] = []
        cursor = 0
        while cursor < len(queue):
            node_id = queue[cursor]
            cursor += 1
            node = all_nodes[node_id]
            if node_id != root_id and not node.executed:
                pending.append(node)
                continue
            for child_id in node.children_ids:
                if child_id in all_nodes:
                    queue.append(child_id)

        if not pending:
            return {}

        return {
            candidate.node_id: self.compute_frontier_score(
                candidate, root_id, all_nodes
            )
            for candidate in pending
        }

    def select_next_node(self, root_id: str, all_nodes: Dict[str, NodeState]) -> Optional[str]:
        """Select the best pending frontier action with lineage-level UCB."""
        scores = self.frontier_scores(root_id, all_nodes)
        if not scores:
            return None

        return max(scores, key=scores.get)
