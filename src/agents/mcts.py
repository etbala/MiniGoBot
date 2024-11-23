import numpy as np
import math
from scipy.special import softmax

from src.main import GoGame

class MCTSNode:
    """A single node in the MCTS search tree."""
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state  # Current board state
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes, indexed by action
        self.visits = 0  # Visit count
        self.value_sum = 0.0  # Sum of backpropagated values
        self.prior = prior  # Prior probability from the policy network
        self.terminal = self.check_terminal()  # Check if state is terminal

    def is_leaf(self):
        """Check if the node is a leaf (no expanded children)."""
        return len(self.children) == 0

    def value(self):
        """Calculate the mean value of this node."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def check_terminal(self):
        """Check if the current state is terminal (game over)."""
        return GoGame.game_ended(self.state)

    def expand(self, action_priors):
        """
        Expand the node by adding child nodes.
        :param action_priors: A list of (action, prior) pairs.
        """
        for action, prior in action_priors:
            if action not in self.children:
                next_state = GoGame.next_state(self.state, action, canonical=True)
                self.children[action] = MCTSNode(state=next_state, parent=self, prior=prior)

    def backprop(self, value):
        """
        Backpropagate the value through the tree.
        :param value: Value to propagate.
        """
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backprop(-value)  # Alternate the perspective


class MCTS:
    """Monte Carlo Tree Search with policy and value guidance."""
    def __init__(self, actor_critic_model, c_puct=1.5, num_simulations=800):
        self.actor_critic_model = actor_critic_model
        self.c_puct = c_puct  # Exploration constant
        self.num_simulations = num_simulations  # Number of simulations

    def run(self, root_state):
        """
        Perform MCTS simulations starting from the root state.
        :param root_state: Initial game state.
        :return: Normalized visit count distribution over actions.
        """
        root = MCTSNode(state=root_state)

        # Initialize root with policy priors
        policy, _ = self.actor_critic_model.pt_actor_critic(root.state[np.newaxis])
        policy = policy.detach().numpy().flatten()
        legal_moves = GoGame.valid_moves(root.state)
        action_priors = [(action, policy[action]) for action in np.where(legal_moves)[0]]
        root.expand(action_priors)

        # Perform simulations
        for _ in range(self.num_simulations):
            self.simulate(root)

        # Compute action probabilities based on visit counts
        visit_counts = np.zeros(GoGame.action_size(root.state))
        for action, child in root.children.items():
            visit_counts[action] = child.visits
        return visit_counts / visit_counts.sum()

    def simulate(self, node):
        """
        Perform a single MCTS simulation.
        :param node: Node to start the simulation from.
        """
        path = []

        # Selection and Expansion
        while not node.is_leaf() and not node.terminal:
            action, node = self.select(node)
            path.append(node)

        # Leaf Evaluation
        if not node.terminal:
            policy, value = self.actor_critic_model.pt_actor_critic(node.state[np.newaxis])
            policy = policy.detach().numpy().flatten()
            value = value.item()

            legal_moves = GoGame.valid_moves(node.state)
            action_priors = [(action, policy[action]) for action in np.where(legal_moves)[0]]
            node.expand(action_priors)
        else:
            value = GoGame.winning(node.state)  # Use the outcome for terminal states

        # Backpropagation
        for node in reversed(path):
            node.backprop(value)
            value = -value  # Alternate perspective

    def select(self, node):
        """
        Select a child node using the UCT formula.
        :param node: Current node.
        :return: Selected action and corresponding child node.
        """
        best_action, best_child = None, None
        max_uct = -float('inf')

        for action, child in node.children.items():
            avg_q = child.value()
            uct_score = avg_q + self.c_puct * child.prior * math.sqrt(node.visits) / (1 + child.visits)
            if uct_score > max_uct:
                best_action, best_child = action, child
                max_uct = uct_score

        return best_action, best_child
