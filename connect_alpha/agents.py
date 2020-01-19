import numpy as np
from tree import Node

class BaseAgent():
    move_count = 0
    def __init__(self):
        pass
     
    def get_last_move(self, prev, curr, conf):
        for i in range(len(prev)):
            if prev[i] == 0 and prev[i] != curr[i]:
                return i % 7

class MctsAgent(BaseAgent):
    def __init__(self, root, mcts):    
        super().__init__(self)
        self.root = root
        self.mcts = mcts
    
    def get_agent(self):
        def agent(obs, conf):
            current_player = obs.mark if obs.mark == 1 else -1
            if sum(obs.board) != 0: 
                BaseAgent.move_count = 0
                self.root = Node(None, np.zeros((6,7)))
                self.mcts.root = self.root
            elif BaseAgent.move_count == 1:
                self.root = Node(None, np.reshape(np.array(obs.board), ( 6, 7)))
            else:
                oppenent_action = self.get_last_move(np.reshape(self.root, (6 * 7)), obs.board, conf)
                self.mcts.move(self.root, )
        return agent

class SelfPlayAgent(MctsAgent):
    def __init__(self, root):
        super().__init__(self, root)

    