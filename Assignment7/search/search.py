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
# Choose some node to expand from the frontier with Stack like implementation


def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    # YOUR CODE HERE
    frontier = util.Stack()
    explored = set()
    initialState = problem.getStartState()
    frontier.push(initialState)
    while not frontier.isEmpty():
        choice = frontier.pop()
        # print choice
        if convertStateToHash(choice) not in explored:
            if problem.isGoalState(choice):
                return choice
            successors = problem.getSuccessors(choice)
            for successor in successors:
                # display(successor[0])
                # print(successor[1])
                frontier.push(successor[0])
            explored.add(convertStateToHash(choice))
    # util.raiseNotDefined()


######################## A-Star and DFS for Map Problem ########################
# Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.
    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    current = problem.G.node[state.state]
    final = problem.G.node[problem.end_node]
    clon = (current['x'], 0, 0)
    clat = (current['y'], 0, 0)
    flon = (final['x'], 0, 0)
    flat = (final['y'], 0, 0)
    hn = util.points2distance((clon, clat), (flon, flat))
    return hn
    # util.raiseNotDefined()


def AStar_search(problem, heuristic=nullHeuristic):
    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """

    def getRoute(node):
        route = []
        route.insert(0, node.state)
        while node.action != -1:
            node = node.parent_node
            route.insert(0, node.state)
        return route

    def function(state):
        h = heuristic(state, problem)
        g = state.path_cost
        return g + h

    frontier = util.PriorityQueueWithFunction(function)
    explored = set()
    start_node = Node(problem.getStartState(), -1, 0.0, -1, 1)
    frontier.push(start_node)
    while not frontier.isEmpty():
        choice = frontier.pop()
        if choice.state not in explored:
            if problem.isGoalState(choice.state):
                return getRoute(choice)
            successors = problem.getSuccessors(choice.state)
            for successor in successors:
                node = Node(successor[0], successor[1], choice.path_cost +
                            successor[2], choice, choice.depth+1)
                frontier.push(node)
            explored.add(choice.state)
    # util.raiseNotDefined()
