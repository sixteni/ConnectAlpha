import numpy as np
from time import sleep, time
import random
from rules import ConnectXRules
from mcts import MCTS, Edge, Node

class NetworkMock():
    def predict(self, state, player):
        sleep(0.002)
        probs = np.random.dirichlet(np.ones(7),size=1)
        return probs[0], random.uniform(-1.0, 1.0)

def run_sim():
    state = np.zeros((6,7))
    root = Node(None, state)
    n = NetworkMock()

    mcts_agent = MCTS(ConnectXRules, n)
    
    mcts_agent.get_best_move(root, 1 , 0)
    

    for action in root.actions:
        print(action.visit_count, end= ' ')
    print()
    print(mcts_agent.winning_moves)

def test():
    pass
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    processes = []
    start = time()
    for i in range(32):
        p = Process(target=run_sim)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    print(time()-start)