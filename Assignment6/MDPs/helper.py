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


class MDP:
    def __init__(self, gridfile, p=1.0):
        gridfile = open(gridfile, 'r')
        lines = gridfile.read().splitlines()
        gridfile.close()

        self.grid = np.array([line.split() for line in lines], dtype=np.int64)

        self.numStates = 0
        self.start = -1
        self.end = -1
        self.numActions = 4  # 0 -> E, 1-> W, 2-> N, 3-> S
        self.discount = 0.99
        self.transitions = []
        l, b = self.grid.shape

        reward = -1
        finalreward = 100000000
        # Stores the mapping from coordinates in grid to state number : List of numbers
        self.state = np.array([[-1]*b]*l, dtype=np.int64)
        # Stores the mapping from each state number to its coordinates in grid : List of tuples
        self.state_index_mapping = []
        self.validMoves = []  # Stores all valid Moves for each state : List of tuples

        for i in range(l):
            for j in range(b):
                if self.grid[i][j] == 0:
                    self.state[i][j] = self.numStates
                    self.state_index_mapping.append((i, j))
                    self.numStates += 1
                elif self.grid[i][j] == 1:
                    pass
                elif self.grid[i][j] == 2:
                    self.state[i][j] = self.numStates
                    self.start = self.numStates
                    self.state_index_mapping.append((i, j))
                    self.numStates += 1
                elif self.grid[i][j] == 3:
                    self.state[i][j] = self.numStates
                    self.end = self.numStates
                    self.state_index_mapping.append((i, j))
                    self.numStates += 1
                else:
                    print("Error: Unknown character in self.grid")

        def addTransitions(i, j):
            """ Assumption: (i, j) is  a valid state
            Add all transitions originating from this state
            Actions 0 -> E, 1-> W, 2-> N, 3-> S
            """
            if self.end != self.state[i][j]:
                validMoves = []
                if i > 0:
                    if ((self.state[i-1][j] != -1) and (self.end == self.state[i-1][j])):
                        validMoves.append((self.state[i-1, j], finalreward, 2))
                    elif (self.state[i-1][j] != -1):
                        validMoves.append((self.state[i-1, j], reward, 2))
                if i < l-1:
                    if ((self.state[i+1][j] != -1) and (self.end == self.state[i+1][j])):
                        validMoves.append((self.state[i+1, j], finalreward, 3))
                    elif (self.state[i+1][j] != -1):
                        validMoves.append((self.state[i+1, j], reward, 3))
                if j > 0:
                    if ((self.state[i][j-1] != -1) and (self.end == self.state[i][j-1])):
                        validMoves.append((self.state[i, j-1], finalreward, 1))
                    elif (self.state[i][j-1] != -1):
                        validMoves.append((self.state[i, j-1], reward, 1))
                if j < b-1:
                    if ((self.state[i][j+1] != -1) and (self.end == self.state[i][j+1])):
                        validMoves.append((self.state[i, j+1], finalreward, 0))
                    elif (self.state[i][j+1] != -1):
                        validMoves.append((self.state[i, j+1], reward, 0))

                self.validMoves.append(validMoves)
                p1 = p + (1.0-p)/len(validMoves)
                p2 = (1.0-p)/len(validMoves)
                validDirections = []
                for move in validMoves:
                    validDirections.append(move[2])
                for move in validMoves:
                    for direction in validDirections:
                        if direction == move[2]:
                            self.transitions.append(Transition(
                                self.state[i][j], move[2], move[0], move[1], p1))
                        elif p2 != 0.0:
                            self.transitions.append(Transition(
                                self.state[i][j], move[2], move[0], move[1], p2))
            else:
                self.validMoves.append([])

        # Adding Transitions
        for state in range(self.numStates):
            i, j = self.state_index_mapping[state]
            addTransitions(i, j)

    def printMDP(self):
        print("numStates", self.numStates)
        print("numActions", self.numActions)
        print("start", self.start)
        print("end", self.end)
        for transition in self.transitions:
            print(transition)
        print("discount", self.discount)
