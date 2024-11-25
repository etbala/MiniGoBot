import numpy as np
import torch
from scipy.special import softmax

class Node:
    def __init__(self, state, parent=None):
        """
        Initialize a Node.
        Args:
            state (np.ndarray): Game state at this node.
            parent (Node, optional): Parent node. Defaults to None.
        """
        self.state = state
        self.parent = parent
        self.child_nodes = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = None

    def is_terminal(self, game):
        """Check if the game is over."""
        return game.game_ended()

    def is_leaf(self):
        """Check if the node is a leaf (no child nodes)."""
        return len(self.child_nodes) == 0

    def expand(self, go_env, policy):
        """
        Expand the node by adding child nodes.
        Args:
            go_env: GymGo environment for generating child states.
            policy (np.ndarray): Policy probabilities for child nodes.
        """
        valid_moves = go_env.valid_moves(self.state)
        child_states = go_env.children(self.state)

        # Iterate through valid moves and child_states simultaneously
        child_state_index = 0
        for move, valid in enumerate(valid_moves):
            if valid:
                # Assign the corresponding child state to the valid move
                self.child_nodes[move] = Node(state=child_states[child_state_index], parent=self)
                self.child_nodes[move].prior = policy[move]
                child_state_index += 1

    def backpropagate(self, value):
        """
        Backpropagate value to parent nodes.
        Args:
            value (float): Value to propagate.
        """
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)  # Invert value for opponent

    def get_value(self, exploration_weight=1.5):
        """
        Compute the UCB1 value for selecting child nodes.
        Args:
            exploration_weight (float): Weight for exploration.
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unexplored nodes
        mean_value = self.value_sum / self.visits
        exploration_term = exploration_weight * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return mean_value + exploration_term

def select(node):
    """
    Select the child node with the highest UCB value.
    Args:
        node (Node): Current node.
    Returns:
        Node: Selected child node.
    """
    return max(node.child_nodes.values(), key=lambda n: n.get_value())

def expand_and_evaluate(node, actor_critic, game):
    """
    Expand the node and evaluate it using the Actor-Critic network.
    Args:
        node (Node): Node to expand.
        actor_critic (callable): Neural network for policy and value estimation.
        game: Game environment for generating child states.
    """
    state = node.state[np.newaxis]  # Add batch dimension
    state = torch.tensor(state, dtype=torch.float32)  # Convert to PyTorch tensor
    policy_logits, value = actor_critic(state)
    policy = softmax(policy_logits.flatten())
    node.expand(game, policy)
    return value.item()

def mcts_step(root, actor_critic, game):
    """
    Perform a single MCTS step.
    Args:
        root (Node): Root node of the search.
        actor_critic (callable): Neural network for policy and value estimation.
        game: Game environment.
    """
    node = root
    while not node.is_leaf() and not node.is_terminal(game):
        node = select(node)

    if node.is_terminal(game):
        value = game.winning(node.state)  # Terminal value
        node.backpropagate(value)
        return

    value = expand_and_evaluate(node, actor_critic, game)
    node.backpropagate(value)

def mcts_search(game, actor_critic, num_simulations):
    """
    Perform MCTS on a given game state.
    Args:
        game: Game environment.
        actor_critic (callable): Neural network for policy and value estimation.
        num_simulations (int): Number of simulations to run.
    Returns:
        Node: Root node after search.
    """
    root_state = game.canonical_state()
    root = Node(state=root_state)

    for _ in range(num_simulations):
        mcts_step(root, actor_critic, game)

    return root
