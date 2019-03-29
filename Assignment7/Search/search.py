import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.
    util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """Search the node that has the lowest combined cost and heuristic first."""

    util.raiseNotDefined()