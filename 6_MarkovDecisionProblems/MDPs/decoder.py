import numpy as np
import sys
from helper import *
import random
import time
import itertools
import bisect

def printAction(path):
    pathstr = []
    for action in path:
        if action == 0:
            pathstr.append("E")
        elif action == 1:
            pathstr.append("W")
        elif action == 2:
            pathstr.append("N")
        elif action == 3:
            pathstr.append("S")
    print(*pathstr, sep=' ', end='\n')

path = []
def foo(valid_moves, current_state, action, p):
    l = len(valid_moves)
    distribution = []
    p1 = p + (1.0 - p)/l
    p2 = (1.0 - p)/l
    for move in valid_moves:
        if move[2] == action:
            distribution.append(p1)
        else:
            distribution.append(p2)
    
    cumdist = list(itertools.accumulate(distribution))
    x = random.random() * cumdist[-1]
    choice = valid_moves[bisect.bisect(cumdist, x)]
    
    state = choice[0]
    action = choice[2]
    path.append(action)
    return state
    

def getPolicy(gridfile, value_and_policy_file, p=1.0):
    mdp = MDP(gridfile)

    v_p_file = open(value_and_policy_file, 'r')
    v_p_lines = v_p_file.read().splitlines()
    v_p_file.close()
    del v_p_lines[-1]
    value_policy = [item.split() for item in v_p_lines]

    current_state = mdp.start
    action = value_policy[current_state][1]

    while action != '-1':  # 0 -> E, 1-> W, 2-> N, 3-> S
        valid_moves = mdp.validMoves[current_state]
        current_state = foo(valid_moves, current_state, int(action), p)
        action = value_policy[current_state][1]
    printAction(path)

randomSeed = 0
random.seed(randomSeed)

gridfile = sys.argv[1]
value_and_policy_file = sys.argv[2]

if (len(sys.argv) == 4):
    p = float(sys.argv[3])
    mdp = MDP(gridfile, p)
    getPolicy(gridfile, value_and_policy_file, p)

else:
    mdp = MDP(gridfile)
    getPolicy(gridfile, value_and_policy_file)
