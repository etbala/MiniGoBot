import numpy as np
import torch
from scipy.special import softmax

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.child_nodes = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = None

    def is_terminal(self, go_env):
        """Check if the node is terminal based on the state."""
        GoGame = go_env.gogame
        return GoGame.game_ended(self.state)

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
        GoGame = go_env.gogame
        GoVars = go_env.govars

        valid_moves = GoGame.valid_moves(self.state)
        child_states = GoGame.children(self.state, canonical=True, padded=True)

        valid_move_indices = np.flatnonzero(valid_moves)

        for move in valid_move_indices:
            child_state = child_states[move]
            self.child_nodes[move] = Node(state=child_state, parent=self)
            self.child_nodes[move].prior = policy[move]

    def backpropagate(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(-value)

    def get_value(self, exploration_weight=1.5):
        if self.visits == 0:
            return float('inf')
        mean_value = self.value_sum / self.visits
        exploration_term = exploration_weight * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return mean_value + exploration_term

    def get_visit_counts(self):
        total_actions = self.state.shape[-1] ** 2 + 1  # Including pass
        visit_counts = np.zeros(total_actions)
        for move, child in self.child_nodes.items():
            visit_counts[move] = child.visits
        return visit_counts


def select(node):
    """
    Select the child node with the highest UCB value.
    Args:
        node (Node): Current node.
    Returns:
        Node: Selected child node.
    """
    return max(node.child_nodes.values(), key=lambda n: n.get_value())

def expand_and_evaluate(node, actor_critic, go_env):
    """
    Expand the node and evaluate it using the Actor-Critic network.
    Args:
        node (Node): Node to expand.
        actor_critic (callable): Neural network for policy and value estimation.
        game: Game environment for generating child states.
    """
    state = node.state[np.newaxis]
    device = next(actor_critic.parameters()).device
    state = torch.tensor(state, dtype=torch.float32).to(device)
    policy_logits, value = actor_critic(state)
    policy_logits = policy_logits.squeeze(0).cpu().detach().numpy()

    # Get valid moves
    GoGame = go_env.gogame
    valid_moves = GoGame.valid_moves(node.state)
    policy_logits[valid_moves == 0] = -np.inf  # Mask invalid moves

    # Apply softmax
    policy = softmax(policy_logits)
    node.expand(go_env, policy)
    return value.item()

def mcts_step(root, actor_critic, go_env):
    """
    Perform a single MCTS step.
    Args:
        root (Node): Root node of the search.
        actor_critic (callable): Neural network for policy and value estimation.
        game: Game environment.
    """
    node = root
    while not node.is_leaf() and not node.is_terminal(go_env):
        node = select(node)

    if node.is_terminal(go_env):
        GoGame = go_env.gogame
        GoVars = go_env.govars
        current_player = GoGame.turn(node.state)
        value = GoGame.winning(node.state)
        # Adjust value to current player's perspective
        if current_player == GoVars.WHITE:
            value = -value
        node.backpropagate(value)
        return

    value = expand_and_evaluate(node, actor_critic, go_env)
    node.backpropagate(value)

def mcts_search(go_env, actor_critic, num_simulations):
    """
    Perform MCTS on a given game state.
    Args:
        game: Game environment.
        actor_critic (callable): Neural network for policy and value estimation.
        num_simulations (int): Number of simulations to run.
    Returns:
        Node: Root node after search.
    """
    state = go_env.canonical_state()
    root = Node(state=state)

    for _ in range(num_simulations):
        mcts_step(root, actor_critic, go_env)

    return root
