# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        
        foodList = newFood.asList()
        
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            score += 1.0 / min(foodDistances) #closer to food
            
        ghostPositions = [currentGameState.getGhostPosition(i+1) for i in range(len(newGhostStates))]
    
        for ghostPos, scaredTime in zip(ghostPositions, newScaredTimes):
            distance = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                score += max(10 - distance, 0)
            else:
                score -= (10 / (distance + 1))

            
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            # Base case: check for win, lose, or maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Get legal actions for the current agent
            legalActions = gameState.getLegalActions(agentIndex)
            
            # Pacman's turn (maximizing player)
            if agentIndex == 0:
                scores = [minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]
                return max(scores) if depth != 0 else legalActions[scores.index(max(scores))]
            
            # Ghosts' turn (minimizing players)
            else:
                nextAgent = agentIndex + 1
                # Check if we need to move to the next depth
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                scores = [minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]
                return min(scores)
        
        # Start the minimax process from Pacman (agentIndex 0) at depth 0
        return minimax(0, 0, gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        alpha, beta = float("-inf"), float("inf")
        
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float("-inf")
            for action in state.getLegalActions(0):
                v = max(v, min_value(state.generateSuccessor(0, action), depth, 1, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
        
        def min_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                if agentIndex == state.getNumAgents() - 1:
                    v = min(v, max_value(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta))
                else:
                    v = min(v, min_value(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v
        
        
        best_score, best_action = float("-inf"), None
        for action in gameState.getLegalActions(0):
            value = min_value(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > best_score:
                best_score, best_action = value, action
            alpha = max(alpha, best_score)
        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)

            if agentIndex == 0:  
                return max_value(state, depth, agentIndex)
            else: 
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents()))
            return v

        def exp_value(state, depth, agentIndex):
            v = 0
            actions = state.getLegalActions(agentIndex)
            probability = 1.0 / len(actions)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                v += probability * expectimax(successor, depth + 1, (agentIndex + 1) % state.getNumAgents())
            return v

        
        best_action = None
        best_score = float("-inf")
        for action in gameState.getLegalActions(0): 
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 1 % gameState.getNumAgents())
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation functio considers following factors,
                  - score from the game state
                  - distance to the closest food
                  - Number of remaining food dots
                  - number of remaining capsules.
                  - ghost proximity and scared timer
    """
    "*** YOUR CODE HERE ***"
    
    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    
    newFood = currentGameState.getFood()
    foodList = newFood.asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList]) if foodList else 0

    ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in ghostStates]
    ghostScore = sum(scaredTime if dist == 0 else -1 / dist for dist, scaredTime in zip(ghostDistances, scaredTimes))

    numCapsules = len(currentGameState.getCapsules())

    evaluationScore = score + (1 / (minFoodDistance + 1)) + ghostScore - 20 * numCapsules - 4 * len(foodList)
    
    return evaluationScore

# Abbreviation
better = betterEvaluationFunction
