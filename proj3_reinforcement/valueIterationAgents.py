# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):
            new_vals = self.values.copy()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    new_vals[state] = max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
            self.values = new_vals

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for t in transitions:
            next_state, probability = t
            reward = self.mdp.getReward(state, action, next_state)
            q_value += probability * (reward + self.discount * self.getValue(next_state))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        best_action, best_val = None, float("-inf")
        for action in self.mdp.getPossibleActions(state):
            q_val = self.getQValue(state, action)
            if q_val > best_val:
                best_action, best_val = action, q_val
        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        iterations_left = self.iterations
        while iterations_left > 0:
            for state in self.mdp.getStates():
                if iterations_left <= 0:
                    return
                if not self.mdp.isTerminal(state):
                    self.values[state] = max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
                iterations_left -= 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        # Let a predecessor be a dict mapping some state to a set of states that can reach it with a non-zero probability
        self.preds = {}
        for state in mdp.getStates():
            for action in mdp.getPossibleActions(state):
                for t in mdp.getTransitionStatesAndProbs(state, action):
                    next_state, probability = t
                    if probability > 0:
                        if next_state not in self.preds:
                            self.preds[next_state] = set()
                        self.preds[next_state].add(state)
        ValueIterationAgent.__init__(self, mdp, discount, iterations)



    def runValueIteration(self):
        q = util.PriorityQueue()
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                max_q = max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
                diff = abs(self.getValue(state) - max_q)
                q.push(state, -diff)

        # Notice that a terminal state is never pushed onto the queue
        # This means we don't ever have to handle terminal states here.
        for _ in range(self.iterations):
            if q.isEmpty():
                return
            state = q.pop()
            self.values[state] = max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
            if state in self.preds:
                for pred in self.preds[state]:
                    max_q = max([self.getQValue(pred, a) for a in self.mdp.getPossibleActions(pred)])
                    diff = abs(self.getValue(pred) - max_q)
                    if diff > self.theta:
                        q.update(pred, -diff)
