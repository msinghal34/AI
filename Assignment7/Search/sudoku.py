# Solve Every Sudoku Puzzle

# See http://norvig.com/sudoku.html

# Throughout this program we have:
# r is a row,    e.g. 'A'
# c is a column, e.g. '3'
# s is a square, e.g. 'A3'
# d is a digit,  e.g. '9'
# u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
# grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
# values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}

import util


def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]


digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
units = dict((s, [u for u in unitlist if s in u])
             for s in squares)
peers = dict((s, set(sum(units[s], []))-set([s]))
             for s in squares)

################ Parse a Grid ################


def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    # To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False  # (Fail if we can't assign d to square s.)
    return values


def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))

################ Constraint Propagation ################


def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  # Already eliminated
    values[s] = values[s].replace(d, '')
    # (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  # Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    # (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  # Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

################ Display as 2-D grid ################


def display(values):
    "Display these values as a 2-D grid."
    width = 1+max(len(values[s]) for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print ''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols)
        if r in 'CF':
            print line
    print

################ Check if all constraints are satisfied ################


def solved(values):
    "A puzzle is solved if each unit is a permutation of the digits 1 to 9."
    def unitsolved(unit): return set(values[s] for s in unit) == set(digits)
    return values is not False and all(unitsolved(unit) for unit in unitlist)

################ Search Problem ################
# You need to modify things below


class SudokuSearchProblem:
    """
    This class outlines the structure of a search problem.
    """

    def __init__(self, values):
        self.start_values = values
        self.nodes_expanded = 0

    def getStartState(self):
        """
        Returns the start state for the search problem.(Dict with key as cell ID and value as possible numbers it can take as string)
        """
        return self.start_values
        # util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        return solved(state)
        # util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor(Dict),
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        Note : Do check if the return dict of assign is not False while calculating the successors
        """
        # Maintain for bookkeeping purposes
        self.nodes_expanded += 1
        # Dont overuse this function since we calculate the nodes expanded using this

        successors = []
        # YOUR CODE HERE
        length = []
        for square in state:
            l = len(state[square])
            if l > 1:
                length.append(l)
        minLength = min(length)
        for square in state:
            if len(state[square]) == minLength:
                choices = state[square]
                for choice in choices:
                    newstate = state.copy()
                    if not (assign(newstate, square, choice) == False):
                        successors.append((newstate, (square, choice), 1))
                break
        return successors
        # util.raiseNotDefined()

# if __name__ == '__main__':
    # You can test your functions and check out the environment here

    # prob1  = '.....6....59.....82....8....45........3........6..3.54...325..6..................'
    # display(parse_grid(prob1))
