import gym
import numpy as np
from scipy import special

from go_bot.data import inverse_canonical_form

GoGame = gym.make('gym_go:go-v0', size=0, disable_env_checker=True).gogame

def get_state_vals(val_func, nodes):
    states = list(map(lambda node: node.state, nodes))
    vals = val_func(np.array(states))
    return vals


def set_state_vals(val_func, nodes):
    vals = get_state_vals(val_func, nodes)
    for val, node in zip(vals, nodes):
        node.set_value(val.item())

    return vals

def invert_vals(vals):
    return -vals

class Node:
    def __init__(self, state, parent=None):
        '''
        Args:
            parent (?Node): parent Node
            prior_value (?float): the state value of this node
            state: state of the game as a numpy array
        '''

        # Go
        self.state = state
        self.child_states = None

        # Links
        self.parent = parent
        self.child_nodes = np.empty(self.actionsize(), dtype=object)

        # Level
        if parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

        # Value
        self.val = None
        self.first_action = None

        # MCT
        self.visits = 0
        self.prior_pi = None
        self.post_vals = []

    def destroy(self):
        for child in self.child_nodes:
            if child is not None:
                child.destroy()
        del self.state
        del self.parent
        del self.child_nodes


    def terminal(self):
        return GoGame.game_ended(self.state)

    def winning(self):
        return GoGame.winning(self.state)

    def isleaf(self):
        # Not the same as whether the state is terminal or not
        return (self.child_nodes == None).all()

    def isroot(self):
        return self.parent is None

    def make_childnode(self, action, state):
        child_node = Node(state, self)
        self.child_nodes[action] = child_node
        if child_node.level == 1:
            child_node.first_action = action
        else:
            assert self.first_action is not None
            child_node.first_action = self.first_action
        return child_node

    def make_children(self):
        # Convert the node's state back to standard form
        state_standard = inverse_canonical_form(self.state)
        # Generate children from the standard state
        child_states_canonical = GoGame.children(state_standard, canonical=True, padded=True)
        actions = np.argwhere(self.valid_moves()).flatten()
        for action in actions:
            child_state_canonical = child_states_canonical[action]
            # The child_state_canonical is already in canonical form
            self.make_childnode(action, child_state_canonical)
        self.child_states = child_states_canonical
        return child_states_canonical

    def get_child_nodes(self):
        real_nodes = list(filter(lambda node: node is not None, self.child_nodes))
        return real_nodes

    def actionsize(self):
        return GoGame.action_size(self.state)

    def valid_moves(self):
        return GoGame.valid_moves(self.state)

    def step(self, move):
        child = self.child_nodes[move]
        if child is not None:
            return child
        else:
            # Convert the node's state back to standard form
            state_standard = inverse_canonical_form(self.state)
            # Generate the next state from the standard state
            next_state_canonical = GoGame.next_state(state_standard, move, canonical=True)
            # Store the child state in canonical form
            child = self.make_childnode(move, next_state_canonical)
            return child


    def set_value(self, val):
        self.val = val

    def get_value(self):
        return self.val

    def inverted_children_values(self):
        inverted_vals = []
        for child in self.child_nodes:
            if child is not None:
                inverted_vals.append(invert_vals(child.val))
            else:
                inverted_vals.append(0)
        return np.array(inverted_vals)


    def backprop(self, val):
        self.post_vals.append(val)
        self.visits += 1
        if self.parent is not None:
            inverted_val = invert_vals(val)
            self.parent.backprop(inverted_val)

    def set_prior_pi(self, prior_pi):
        if prior_pi is not None:
            self.prior_pi = prior_pi
        else:
            # Uses children state values to make prior pi
            self.prior_pi = np.zeros(self.actionsize())
            valid_moves = self.valid_moves()
            where_valid = np.argwhere(valid_moves).flatten()
            q_logits = self.inverted_children_values()
            self.prior_pi[where_valid] = special.softmax(q_logits[where_valid])

            assert not np.isnan(self.prior_pi).any()

    def get_visit_counts(self):
        move_visits = []
        for child in self.child_nodes:
            if child is None:
                move_visits.append(0)
            else:
                move_visits.append(child.visits)
        return np.array(move_visits)

    def get_ucbs(self):
        ucbs = np.full(self.actionsize(), np.nan, dtype=np.float32)
        valid_moves = np.argwhere(self.valid_moves()).flatten()
        for a in valid_moves:
            avg_q, n = 0, 0
            prior_q = self.prior_pi[a]
            child = self.child_nodes[a]
            if child is not None and child.visits > 0:
                n = child.visits
                assert len(child.post_vals) > 0, (child.post_vals, n)
                avg_q = invert_vals(np.mean(np.tanh(child.post_vals)))

            u = 1.5 * prior_q * np.sqrt(self.visits) / (1 + n)
            ucbs[a] = avg_q + u
        return np.array(ucbs)

    def __str__(self):
        result = ''
        if self.val is not None:
            result += f'{self.val:.2f}V'
        if len(self.post_vals) > 0:
            result += f' {np.mean(self.post_vals):.2f}AV'

        result += f' {self.level}L {self.visits}N'

        return result


def find_next_node(node):
    curr = node
    while curr.visits > 0 and not curr.terminal():
        ucbs = curr.get_ucbs()
        move = np.nanargmax(ucbs)
        curr = curr.step(move)

    return curr


def mcts_step(rootnode, actor_critic=None, critic=None):
    # Next node to expand
    node = find_next_node(rootnode)

    # Convert state to canonical form for the neural network
    state_canonical = GoGame.canonical_form(node.state)

    # Initialize pi_logits to None
    pi_logits = None

    # Compute values on internal nodes
    if actor_critic is not None:
        pi_logits, val_logits = actor_critic(state_canonical[np.newaxis])
    else:
        val_logits = critic(state_canonical[np.newaxis])

    # Backpropagate the value
    node.backprop(val_logits.item())

    # If terminal, no need to expand further
    if node.terminal():
        return

    # Set prior probabilities
    if pi_logits is not None:
        pi = special.softmax(pi_logits.flatten())
        node.set_prior_pi(pi)
    else:
        node.make_children()
        next_nodes = node.get_child_nodes()
        # Set values for child nodes using the critic
        for child in next_nodes:
            child_state_canonical = GoGame.canonical_form(child.state)
            val = critic(child_state_canonical[np.newaxis])
            child.set_value(val.item())
        node.set_prior_pi(None)



def mcts_search(go_env, num_searches, actor_critic=None, critic=None):
    # Setup the root
    rootstate = go_env.state()
    rootnode = Node(rootstate)

    # The first iteration doesn't count towards the number of searches
    mcts_step(rootnode, actor_critic, critic)

    # MCT Search
    for i in range(0, num_searches):
        mcts_step(rootnode, actor_critic, critic)

    return rootnode
