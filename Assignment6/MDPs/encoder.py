import numpy as np
import sys
from helper import *

gridfile = sys.argv[1]
if (len(sys.argv) == 3):
    p = float(sys.argv[2])
    mdp = MDP(gridfile, p)
    mdp.printMDP()
else:
    mdp = MDP(gridfile)
    mdp.printMDP()
