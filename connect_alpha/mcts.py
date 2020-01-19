from tree import Node, Edge

class MCTS(): 
    
    def __init__(self, game_rules, network):
        self.root = None
        self.game_rules = game_rules
        self.network = network
        self.itterations_per_move = 400
        self.c_puct = 4
        self.tau = 1
        self.winning_moves = 0

    def get_best_move(self, root, player, move_count):
        self.root = root 
        self.itterations = self.itterations_per_move
        self.move_count = move_count
        self.root_player = player
        self.winning_moves = 0
        while self.itterations > 0: 
            move_count = self.move_count       
            (leaf, player, move_count) = self.traverse_tree(self.root, self.root_player, move_count)                       
            v = self.expand_and_evaluate(leaf, player, move_count)             
            self.backpropagate(leaf, -v) 
            self.itterations -= 1
        if self.move_count > 10:
          self.tau = 0.1
        pi = self.get_policy_vector(self.root)        
        action = pi.index(max(pi))   

        return (action, pi)

    def move(self, action, player, move_count):

        actions_aviable = [a.action for a in self.root.actions]
        
        if action not in actions_aviable:           
           self.expand_and_evaluate(self.root, player, move_count)
           actions_aviable = [a.action for a in self.root.actions]
        action_index = actions_aviable.index(action)
        self.root = self.root.actions[action_index].child_state

        return self.root

  
    def traverse_tree(self, node, player, move_count):
        '''
            Traverses the tree by selecting the action with the highest
            upper confidence bound until the node is a leaf node (e.g. has no children),
            due to either the node has not yet been evaluated and expanded or the node is a terminal node
            (i.e. ends the game with a win, loss or draw)

            returns the node and the current player (1/-1): (Node, int)
        '''

        while not node.is_leaf_node():            
            node = node.select_best_child(self.c_puct)                
            player *= -1
            move_count += 1           

        return node, player, move_count
     
    # function for backpropagation 
    def backpropagate(self, node, v): 
        if node == self.root: 
            return
        node = self.update_stats(node, v)  
        self.backpropagate(node , -v * 0.95) 
    
    def update_stats(self, node, v):
        '''
            Incrementes the visit count to: N(st, at) = N(st, at) + 1, 
            Updates the total action value to: W(s, a) = W(s, a) + v, 
            Updates the mean action value to: Q(s, a) = W(s,a) / N(s, a)
        '''
        action = node.parent_action
        action.total_action_value += v
        action.visit_count += 1
        action.mean_action_value = action.total_action_value / action.visit_count
        parent_node = action.parent_state
        parent_node.visit_count += 1
        if (v <= -0.95):
            for a in parent_node.actions:
                a.prior_probability = 0 if action.action != a.action else 1
        return parent_node

    def expand_and_evaluate(self, node, player, move_count):
        if node.parent_action is not None:
          last_move = node.parent_action.action
          (is_terminal, value) = self.game_rules.is_terminal_state(node.state, last_move, -player, move_count)
          if is_terminal:
              node.is_terminal = is_terminal
              node.terminal_value = value
              self.winning_moves += 1
              return value

        (p, v) = self.network.predict(node.state, player)
        
        # Returns a one dim list of indexes of legal actions
        legal_actions = self.game_rules.get_legal_actions(node.state)
        
        # Remove the illegal actions
        p = p[legal_actions]
        
        # Normalize the the legal actions
        p_sum, p_min = p.sum(), p.min()
        p = (p - p_min) / p_sum
        
        i = 0
        
        for prob in p:
            action = Edge(node, prob, legal_actions[i])
            state = Node(action, self.game_rules.make_move(node.state, legal_actions[i], player))
            action.child_state = state
            node.actions.append(action)
            i += 1
        return v

    def get_policy_vector(self, node):
        '''
             π(a|s0) = (N(s0, a) ^ 1/τ) / (Pb N(s0, b) ^ 1/τ)

             returns the policy vector π
        '''
        pi = [ -1.0 for _ in range(self.game_rules.COLLS) ]
        legal_actions = self.game_rules.get_legal_actions(node.state)
        for action in node.actions:
            if action.action in legal_actions:
                value = pow(action.visit_count, 1 / self.tau) / pow(node.visit_count, 1 / self.tau)
                pi[action.action] = value
        return pi