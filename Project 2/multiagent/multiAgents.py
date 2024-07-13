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
        #return successorGameState.getScore()

        score = successorGameState.getScore()
        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / closestFoodDist
        
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

    def minimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return [self.evaluationFunction(gameState)]
        
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth + 1
        
        results = []
        
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            #print(successor)
            result = self.minimax(successor, nextDepth, nextAgent)
            results.append(result[0])
        
        if agentIndex == 0:
            maxValue = max(results)
            return [maxValue] + [action for action, value in zip(gameState.getLegalActions(agentIndex), results) if value == maxValue]
        else:
            minValue = min(results)
            return [minValue]

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
        # util.raiseNotDefined()
        result = self.minimax(gameState, 0, 0)
        return random.choice(result[1:])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBetaSearch(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents

        if agentIndex == 0:  
            value = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.alphaBetaSearch(successor, depth, nextAgent, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
        else:  
            value = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if nextAgent == 0:
                    nextDepth = depth - 1
                else:
                    nextDepth = depth
                value = min(value, self.alphaBetaSearch(successor, nextDepth, nextAgent, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        actions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.alphaBetaSearch(successor, self.depth, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth if nextAgent != 0 else depth + 1

        if agentIndex == 0:  
            value = float("-inf")
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, self.expectimax(successor, nextDepth, nextAgent))
            return value
        else:  
            value = 0
            legalActions = gameState.getLegalActions(agentIndex)
            probability = 1.0 / len(legalActions)
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value += self.expectimax(successor, nextDepth, nextAgent) * probability
            return value

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        bestAction = None
        bestValue = float("-inf")
        
        for action in gameState.getLegalActions(0):  
            successor = gameState.generateSuccessor(0, action)
            value = self.expectimax(successor, 0, 1)  
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    minFoodDist = float('inf')
    minGhostDist = float('inf')
    
    for foodPos in food.asList():
        minFoodDist = min(minFoodDist, manhattanDistance(pacmanPos, foodPos))

    for ghostState in ghostStates:
        if ghostState.scaredTimer == 0:
            minGhostDist = min(minGhostDist, manhattanDistance(pacmanPos, ghostState.getPosition()))

    remainingFood = currentGameState.getNumFood()
    currentScore = currentGameState.getScore()

    if minFoodDist == 0: minFoodDist = 1
    if minGhostDist == 0: minGhostDist = 1

    return currentScore + 1.0/minFoodDist - 2.0/minGhostDist - 0.5*remainingFood

# Abbreviation
better = betterEvaluationFunction