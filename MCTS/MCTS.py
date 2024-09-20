from MCTS.TreeNode import TreeNode


class MCTSSearch:
    def __init__(self, root_state):
        """
        Initializes the MCTS search.

        Args:
            root_state: The initial state of the game or problem.
        """
        self.root = TreeNode(root_state)
        self.training_data = []

    def search_anet_rollout(self, iterations, anet):
        """
        Perform the MCTS search over a specified number of iterations, using a blend of ANET and random rollouts.

        Args:
            iterations (int): The base number of iterations to run the search.
            anet (ANET): The neural network model to use for action selection during rollouts.
        Returns:
            The best action determined by the search.
        """
        for _ in range(iterations):
            node = self.select_node_for_expansion()
            expanded_node = self.expand_node(node)
            result = self.simulate_rollout_with_anet(expanded_node if expanded_node else node, anet)
            self.backpropagate_result(node, result)

        return self.select_best_action_from_root()

    def simulate_rollout_with_anet(self, node, anet):
        """
        Simulate a rollout using the ANET model to select actions, with a dynamic chance of applying the ANET directly to the current state.
        """
        state = node.state
        while not state.is_terminal():
            possible_actions = state.get_possible_actions()
            state_tensor = state.get_state_tensor()

            best_action, _ = anet.predict(state_tensor.unsqueeze(0), possible_actions, stochastic=True)

            state = state.take_action(best_action)

        return state.get_result(self.root.state.current_player)

    def select_node_for_expansion(self):
        """
        Selects a node for expansion using the tree policy.
        """
        node = self.root
        while node.children:
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def expand_node(self, node):
        """
        Expands the selected node by adding a new child.

        Args:
            node: The node to expand.
        """
        if not node.is_fully_expanded() and not node.is_terminal():
            return node.expand()
        return None

    def backpropagate_result(self, node, result):
        """
        Backpropagate the result of the simulation up the tree.

        Args:
            node (TreeNode): The node from which to start backpropagation.
            result: The result of the simulation.
        """
        while node:
            node.update(result)
            node = node.parent

    def select_best_action_from_root(self):
        """
        Select the best action from the root node based on the win rate.

        Returns:
            The best action determined by the MCTS.
        """
        best_child = max(self.root.children,
                         key=lambda x: x.visits if x.visits > 0 else 0,
                         default=None)

        return best_child.state.get_action()

    def get_visit_counts(self, node):
        """
        Compute the normalized visit counts for all possible actions from the root node's state.

        Args:
            node (TreeNode): The node for which to compute visit counts.

        Returns:
            dict: A dictionary mapping actions to their normalized visit counts.
        """
        root_actions = node.state.get_all_actions()
        visit_counts = {action: 0 for action in root_actions}
        total_visits = sum(child.visits for child in node.children)

        for child in node.children:
            action = child.state.get_action()
            if action in root_actions:
                visit_counts[action] = child.visits / total_visits if total_visits > 0 else 0

        return visit_counts

    def collect_training_data(self):
        """
        Collects the training data for the neural network.
        """
        state_vector = self.root.state.get_state_tensor()
        visit_counts = self.get_visit_counts(self.root)

        return state_vector, visit_counts