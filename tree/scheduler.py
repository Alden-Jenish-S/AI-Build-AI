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

    def get_exploration_constant(self, t: int) -> float:
        """Computes exploration constant c_t based on piecewise decay formula."""
        t1 = int(self.p1 * self.total_budget)
        t2 = int(self.p2 * self.total_budget)
        
        if t < t1:
            return self.c_0
        elif t <= t2:
            return max(self.c_0 - self.alpha * (t - t1), self.c_min)
        else:
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

    def select_next_node(self, root_id: str, all_nodes: Dict[str, NodeState]) -> str:
        """Selects the next node to expand/evaluate by recursively walking down the tree 
        following the maximum UCB1 paths (multi-armed bandit selection).
        Skips nodes that have already been executed.
        """
        curr_node = all_nodes[root_id]
        t = self.current_step
        
        while curr_node.children_ids:
            best_child = None
            best_score = -float('inf')
            
            for cid in curr_node.children_ids:
                child = all_nodes[cid]
                # Skip nodes that have already been fully executed
                # UNLESS they have unexecuted children (tree must be traversable)
                if child.executed and not child.children_ids:
                    continue
                    
                score = self.compute_ucb_score(child, curr_node.visits, t)
                if score > best_score:
                    best_score = score
                    best_child = child
                    
            if best_child is None:
                # All children are executed leaves — return current node
                # (this branch is exhausted)
                break
                
            curr_node = best_child
            
            # If this node is executed but has children, keep descending
            # If it's unexecuted, stop — this is our selection
            if not curr_node.executed:
                break
            
        return curr_node.node_id
