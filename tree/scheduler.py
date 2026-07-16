import math
from typing import Dict, List, Optional
from .node import NodeState

class UCB1Scheduler:
    def __init__(self, total_budget: int, c_0: float = 1.414, c_min: float = 0.5, 
                 alpha: float = 0.01, p1: float = 0.3, p2: float = 0.7):
        self.total_budget = total_budget
        self.c_0 = c_0
        self.c_min = c_min
        self.alpha = alpha
        self.p1 = p1
        self.p2 = p2
        self.current_step = 0
        self.warmup_budget = 0

    def set_warmup_budget(self, warmup_budget: int) -> None:
        """Exclude forced initial coverage from UCB decay and visit accounting."""
        self.warmup_budget = max(0, min(int(warmup_budget), self.total_budget))

    def _eligible_step(self, t: int) -> int:
        return max(0, int(t) - self.warmup_budget)

    def _eligible_budget(self) -> int:
        return max(1, self.total_budget - self.warmup_budget)

    def get_exploration_constant(self, t: int) -> float:
        """Compute decay over UCB-eligible experiments, excluding forced warm-up."""
        eligible_budget = self._eligible_budget()
        eligible_t = self._eligible_step(t)
        t1 = self.p1 * eligible_budget
        t2 = self.p2 * eligible_budget
        
        if eligible_t < t1:
            return self.c_0
        if eligible_t <= t2 and t2 > t1:
            progress = (eligible_t - t1) / (t2 - t1)
            return self.c_0 - progress * (self.c_0 - self.c_min)
        return self.c_min

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
    def _depth(node_id: str, all_nodes: Dict[str, NodeState]) -> int:
        depth = 0
        current = all_nodes[node_id]
        while current.parent_id is not None:
            depth += 1
            current = all_nodes[current.parent_id]
        return depth

    @staticmethod
    def _root_branch(node_id: str, root_id: str, all_nodes: Dict[str, NodeState]) -> NodeState:
        current = all_nodes[node_id]
        while current.parent_id not in {None, root_id}:
            current = all_nodes[current.parent_id]
        return current

    @staticmethod
    def _nearest_measured_reward(
        node_id: str, all_nodes: Dict[str, NodeState]
    ) -> Optional[float]:
        current = all_nodes[node_id]
        while current.parent_id is not None:
            current = all_nodes[current.parent_id]
            if current.node_type != "implementation" or not current.result:
                continue
            reward = current.result.get("reward")
            if isinstance(reward, (int, float)) and math.isfinite(reward):
                return float(reward)
        return None

    @staticmethod
    def _has_implementation_ancestor(
        node_id: str, all_nodes: Dict[str, NodeState]
    ) -> bool:
        current = all_nodes[node_id]
        while current.parent_id is not None:
            current = all_nodes[current.parent_id]
            if current.node_type == "implementation":
                return True
        return False

    def compute_frontier_score(
        self,
        node: NodeState,
        root_id: str,
        all_nodes: Dict[str, NodeState],
    ) -> float:
        """Score a pending experiment or planning action.

        UCB is applied to the top-level solution lineage, while the nearest measured
        ancestor supplies an exploitation prior. This makes the scheduler useful for
        trees whose pending nodes have not themselves been visited yet.
        """
        root = all_nodes[root_id]
        branch = self._root_branch(node.node_id, root_id, all_nodes)
        branch_mean = branch.total_reward / branch.visits if branch.visits else 0.0
        measured_reward = self._nearest_measured_reward(node.node_id, all_nodes)
        exploitation = measured_reward if measured_reward is not None else branch_mean
        eligible_root_visits = max(root.visits - self.warmup_budget, 0)
        # Every root branch receives at most one forced screening visit. Remove it
        # before comparing how much UCB-eligible attention each lineage received.
        eligible_branch_visits = max(branch.visits - (1 if branch.visits else 0), 0)
        exploration = self.get_exploration_constant(self.current_step) * math.sqrt(
            math.log(eligible_root_visits + 2.0) / (eligible_branch_visits + 1.0)
        )

        config = node.config or {}
        prior = float(config.get("priority", 0.0) or 0.0)
        # Plan all initial approaches before spending experiment budget, then favor
        # runnable implementations over speculative follow-up planning actions.
        if (
            measured_reward is None
            and node.node_type == "technique"
            and not self._has_implementation_ancestor(node.node_id, all_nodes)
        ):
            type_bonus = 0.15
        elif node.node_type == "implementation":
            type_bonus = 0.10
        else:
            type_bonus = 0.02
        fidelity_bonus = {"screen": 0.0, "medium": 0.015, "full": 0.03}.get(
            node.fidelity, 0.0
        )
        depth_penalty = 0.005 * self._depth(node.node_id, all_nodes)
        return exploitation + exploration + prior + type_bonus + fidelity_bonus - depth_penalty

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
