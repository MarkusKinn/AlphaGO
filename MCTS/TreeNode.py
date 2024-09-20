import math

class TreeNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree. Each node maintains its own state,
    parent, children, and statistics relevant for MCTS operations.
    """
    def __init__(self, state, parent=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.children = []
        self.total_action_value = 0
        self.visits = 0
        self.untried_actions = self.state.get_possible_actions()
        self.player = self.state.current_player
        self.prior_prob = prior_prob

    def expand(self):
        """
        Expand the node by adding a new child node.

        Returns:
            TreeNode: The newly created child node.
        """
        action = self.untried_actions.pop()
        new_state = self.state.take_action(action)
        child_node = TreeNode(new_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """
        Update the node with the result of a simulation.

        Args:
            result (float): The result of the simulation.
        """
        self.visits += 1
        self.total_action_value += result if self.player == self.state.current_player else 1-result

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded.

        Returns:
            bool: True if all possible actions have been explored, False otherwise.
        """
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """
        Check if the node's state is a terminal state.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return self.state.is_terminal()

    def best_child(self, c_param=1.5):
        """
        Select the best child using the Upper Confidence Bound (UCB) algorithm.

        Args:
            c_param (float): Exploration parameter.

        Returns:
            TreeNode: The best child node for exploration.
        """
        return max(self.children, key=lambda child: child.ucb_score(c_param), default=None)

    def ucb_score(self, c_param):
        """
        Calculate the Upper Confidence Bound (UCB) score for this node based on AlphaGo's tree policy.

        Args:
            c_param (float): Exploration parameter.

        Returns:
            float: The UCB score.
        """
        if self.visits == 0:
            return float('inf')  # Infinite score for unvisited nodes to encourage exploration

        avg_action_value = self.total_action_value / self.visits
        exploration_term = c_param * self.prior_prob * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_action_value + exploration_term
