# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    actions = []
    closed = set()
    fringe = util.Stack()
    fringe.push(problem.getStartState())
    return depthFirstSearchHelper(problem, closed, fringe, actions)

def depthFirstSearchHelper(problem, closed, fringe, actions):
    curr_state = fringe.pop()
    if problem.isGoalState(curr_state):
        return actions
    if curr_state not in closed:
        closed.add(curr_state)
        for succ, action, stepCost in problem.getSuccessors(curr_state):
            fringe.push(succ)
            actions_new = actions.copy()
            actions_new.append(action)
            res = depthFirstSearchHelper(problem, closed, fringe, actions_new)
            if res is not None:
                return res

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    closed = set()
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))
    while fringe:
        curr_state, action_state = fringe.pop()
        if problem.isGoalState(curr_state):
            return action_state
        closed.add(curr_state)
        for succ, action, stepCost in problem.getSuccessors(curr_state):
            if succ not in closed:
                closed.add(succ)
                action_state_cp = action_state.copy()
                action_state_cp.append(action)
                fringe.push((succ, action_state_cp))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    closed = set()
    fringe = util.PriorityQueueWithFunction(lambda x: x[0])
    fringe.push((0, problem.getStartState(), []))
    while fringe:
        cost, curr_state, action_state = fringe.pop()
        if problem.isGoalState(curr_state):
            return action_state
        if curr_state not in closed:
            closed.add(curr_state)
            for succ, action, stepCost in problem.getSuccessors(curr_state):
                action_state_cp = action_state.copy()
                action_state_cp.append(action)
                fringe.push((cost + stepCost, succ, action_state_cp))


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    closed = {}
    fringe = util.PriorityQueueWithFunction(lambda x: x[0] + heuristic(x[1], problem))
    fringe.push((0, problem.getStartState(), [])) #first 0: stepCost + heuristic
    while fringe:
        curr_cost, curr_state, action_state = fringe.pop()
        if problem.isGoalState(curr_state):
            return action_state
        if curr_state not in closed or curr_cost < closed[curr_state]:
            closed[curr_state] = curr_cost
            for succ, action, stepCost in problem.getSuccessors(curr_state):
                action_state_cp = action_state.copy()
                action_state_cp.append(action)
                total_cost = curr_cost + stepCost
                fringe.push((total_cost, succ, action_state_cp))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
