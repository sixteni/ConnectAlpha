import math

class Node():
    def __init__(self, parent_action, state):
        
        # Actions (edges) (s, a) for all legal actions a ∈ A(s)
        self.actions = []

        # N(s, b) the sum of visit count for all actions below this state 
        self.visit_count = 0

        # Represents the board state
        self.state = state

        # Represents parent action (edge)
        self.parent_action = parent_action

        self.is_terminal = False
        self.terminal_value = -1

    def is_leaf_node(self):
        '''
            Checks if a node has been expanded
            
            returns: True/False
        '''
        return len(self.actions) == 0

    def select_best_child(self, c_puct):
        '''
            Selects the action with the maximum upper confidence bound
            or in this case the polynomial upper confidence trees (PUCT)
            a(t) = argmax(Q(s(t) , a) + U(s(t) , a))
            where U(s, a) = c_puct * P(s,a) * sqr(sum(N(s,b))) / 1 + N(s,a)

            returns child node of best action: Node
        '''
        ucb_list = []
        for action in self.actions:
            u = c_puct * action.prior_probability * math.sqrt(self.visit_count) / (1 + action.visit_count)
            ucb = action.mean_action_value + u
            ucb_list.append(ucb)
        child_index = ucb_list.index(max(ucb_list))
        return self.actions[child_index].child_state

class Edge():
    def __init__(self, parent_state, prior_probability, action):

        # The parent state
        self.parent_state = parent_state

        # P(s, a) the prior probability of selecting this edge
        self.prior_probability = prior_probability

        # The action/move
        self.action = action

        # Q(s, a) the mean action-value i.e. the mean win probability
        self.mean_action_value = 0.0

        # W(s, a) the total action-value i.e. the win probability 
        # for being in this state for the curren player
        self.total_action_value = 0.0

        # N(s, a) the visit count
        self.visit_count = 0

        # The child state
        self.child_state = None