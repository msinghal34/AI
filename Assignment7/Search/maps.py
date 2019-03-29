import networkx as nx
import osmnx as ox
import util
import itertools

class MapSearchProblem:
    """
    This class outlines the structure of a search problem
    """

    def  __init__(self, G, start_node, end_node):
        self.G = G
        self.start_node = start_node
        self.end_node = end_node
        self.nodes_expanded = 0

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, node):
        """
        node: Search state
        Returns True if node is the goal state otherwise False
        """
        util.raiseNotDefined()

    def getSuccessors(self, node):
        """
        node: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        ## Maintain for bookkeeping purposes    
        self.nodes_expanded += 1 
        ## Dont overuse this function since we calculate the nodes expanded using this

        successors = []
        ## YOUR CODE HERE
        util.raiseNotDefined()

        return successors

if __name__ == "__main__":

    pass