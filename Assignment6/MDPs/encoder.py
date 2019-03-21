import numpy as np


class Transition:
    def __init__(self, s1, ac, s2, r, p):
        self.s1 = s1
        self.ac = ac
        self.s2 = s2
        self.r = r
        self.p = p

    def __repr__(self):
        return str("transition" + " " + str(self.s1) + " " + str(self.ac) + " " + str(self.s2) + " " + str(self.r) + " " + str(self.p))


def createMDP(gridfile, mdpfile):
    gridfile = open(gridfile, 'r')
    lines = gridfile.read().splitlines()
    gridfile.close()

    grid = np.array([line.split() for line in lines], dtype=np.int64)

    numStates = 0
    start = -1
    end = -1
    numActions = 4  # 0 -> E, 1-> W, 2-> N, 3-> S
    discount = 0.99
    transitions = []
    l, b = grid.shape

    reward = -1
    finalreward = 100000000
    state = np.array([[-1]*b]*l, dtype=np.int64)

    def addTransitions(i, j):
        """ Assumption: (i, j) is  a valid state
        Add all transitions related to left and top states if valid
        """
        if i > 0:
            if state[i-1][j] != -1 and end == state[i][j]:
                transitions.append(Transition(
                    state[i-1][j], 3, state[i][j], finalreward, 1))
            elif state[i-1][j] != -1:
                transitions.append(Transition(
                    state[i-1][j], 3, state[i][j], reward, 1))
                transitions.append(Transition(
                    state[i][j], 2, state[i-1][j], reward, 1))
        if j > 0:
            if state[i][j-1] != -1 and end == state[i][j]:
                transitions.append(Transition(
                    state[i][j-1], 0, state[i][j], finalreward, 1))
            elif state[i][j-1] != -1:
                transitions.append(Transition(
                    state[i][j-1], 0, state[i][j], reward, 1))
                transitions.append(Transition(
                    state[i][j], 1, state[i][j-1], reward, 1))

    for i in range(l):
        for j in range(b):
            if grid[i][j] == 0:
                state[i][j] = numStates
                addTransitions(i, j)
                numStates += 1
            elif grid[i][j] == 1:
                pass
            elif grid[i][j] == 2:
                state[i][j] = numStates
                start = numStates
                addTransitions(i, j)
                numStates += 1
            elif grid[i][j] == 3:
                state[i][j] = numStates
                end = numStates
                addTransitions(i, j)
                numStates += 1
            else:
                print("Error: Unknown character in grid")

    mdpfile = open(mdpfile, 'w')
    print("numStates", numStates, file=mdpfile)
    print("numActions", numActions, file=mdpfile)
    print("start", start, file=mdpfile)
    print("end", end, file=mdpfile)
    for transition in transitions:
        print(transition, file=mdpfile)
    print("discount", discount, file=mdpfile)
    mdpfile.close()


gridfile = input("GridFile: ")
mdpfile = input("MDPfile: ")
createMDP(gridfile, mdpfile)
