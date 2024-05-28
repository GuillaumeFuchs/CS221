from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

# BEGIN_HIDE
from queue import Queue

# END_HIDE


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        Description of GameState and helper functions:

        A GameState specifies the full game state, including the food, capsules,
        agent configurations and score changes. In this function, the |gameState| argument
        is an object of GameState class. Following are a few of the helper methods that you
        can use to query a GameState object to gather information about the present state
        of Pac-Man, the ghosts and the maze.

        gameState.getLegalActions():
            Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

        gameState.generateSuccessor(agentIndex, action):
            Returns the successor state after the specified agent takes the action.
            Pac-Man is always agent 0.

        gameState.getPacmanState():
            Returns an AgentState object for pacman (in game.py)
            state.configuration.pos gives the current position
            state.direction gives the travel vector

        gameState.getGhostStates():
            Returns list of AgentState objects for the ghosts

        gameState.getNumAgents():
            Returns the total number of agents in the game

        gameState.getScore():
            Returns the score corresponding to the current state of the game


        The GameState class is defined in pacman.py and you might want to look into that for
        other helper methods, though you don't need to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        # BEGIN_HIDE
        # END_HIDE

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # BEGIN_HIDE
        # END_HIDE
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
          The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game

        gameState.getScore():
          Returns the score corresponding to the current state of the game

        gameState.isWin():
          Returns True if it's a winning state

        gameState.isLose():
          Returns True if it's a losing state

        self.depth:
          The depth to which search should continue

        """


pass
# ### START CODE HERE ###


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction. Terminal states can be found by one of the following:
        pacman won, pacman lost or there are no legal moves.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

        Directions.STOP:
          The stop direction, which is always legal

        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
          Returns the total number of agents in the game

        gameState.getScore():
          Returns the score corresponding to the current state of the game

        gameState.isWin():
          Returns True if it's a winning state

        gameState.isLose():
          Returns True if it's a losing state

        self.depth:
          The depth to which search should continue

        """

        # ### YOUR CODE HERE # ###
        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            if agentIndex == 0:  # Maximize for Pac-Man
                return maxValue(state, agentIndex, depth)
            else:  # Minimize for ghosts
                return minValue(state, agentIndex, depth)

        def maxValue(state, agentIndex, depth):
            v = float("-inf")
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(
                    v, value(successor, (agentIndex + 1) % numAgents, depth)
                )  # Alternate between players
            return v

        def minValue(state, agentIndex, depth):
            v = float("inf")
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1:  # If last ghost
                    v = min(
                        v, value(successor, 0, depth - 1)
                    )  # Decrease depth for Pac-Man
                else:
                    v = min(v, value(successor, agentIndex + 1, depth))  # Next ghost
            return v

        numAgents = gameState.getNumAgents()
        legalActions = gameState.getLegalActions(0)  # Pac-Man's legal actions
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            v = value(successor, 1, self.depth)  # Start with first ghost at depth
            if v > bestValue:
                bestValue = v
                bestAction = action
        return bestAction

    # ### END CODE HERE ###


######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (problem 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # pass

        # ### START CODE HERE ###
        def maxValue(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v = float("-inf")
            bestActions = []
            legalActions = state.getLegalActions(0)  # assuming Pacman is agent 0
            if not legalActions:
                return self.evaluationFunction(state), None
            for action in legalActions:
                successor = state.generateSuccessor(0, action)
                newValue, _ = minValue(successor, depth, 1, alpha, beta)
                if newValue > v:
                    v = newValue
                    bestActions = [action]
                elif newValue == v:
                    bestActions.append(action)
                if v > beta:
                    return v, random.choice(
                        bestActions
                    )  # Choose random action when tie
                alpha = max(alpha, v)
            return v, random.choice(bestActions)  # Choose random action when tie

        def minValue(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v = float("inf")
            bestActions = []
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state), None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    newValue, _ = maxValue(successor, depth - 1, alpha, beta)
                else:
                    newValue, _ = minValue(
                        successor, depth, agentIndex + 1, alpha, beta
                    )
                if newValue < v:
                    v = newValue
                    bestActions = [action]
                elif newValue == v:
                    bestActions.append(action)
                if v < alpha:
                    return v, random.choice(
                        bestActions
                    )  # Choose random action when tie
                beta = min(beta, v)
            return v, random.choice(bestActions)  # Choose random action when tie

        _, action = maxValue(gameState, self.depth, float("-inf"), float("inf"))
        return action

        # ### END CODE HERE ###


######################################################################################
# Problem 3b: implementing expectimax


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        # pass
        # ### START CODE HERE ###
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            if agentIndex == 0:
                return max_value(state, depth)
            else:
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth):
            v = float("-inf")
            best_action = None
            legal_actions = state.getLegalActions(0)
            for action in legal_actions:
                successor = state.generateSuccessor(0, action)
                value, _ = expectimax(successor, depth, 1)
                if value > v:
                    v = value
                    best_action = action
            return v, best_action

        def exp_value(state, depth, agentIndex):
            v = 0
            legal_actions = state.getLegalActions(agentIndex)
            probability = 1.0 / len(legal_actions)
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value, _ = expectimax(successor, depth - 1, 0)
                else:
                    value, _ = expectimax(successor, depth, agentIndex + 1)
                v += probability * value
            return v, None

        _, action = expectimax(gameState, self.depth, 0)
        return action
        # ### END CODE HERE ###


######################################################################################
# Problem 4a (extra credit): creating a better evaluation function
"""
def betterEvaluationFunction(currentGameState):
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    ghostStates = currentGameState.getGhostStates()
    closestFoodDist = min([util.manhattanDistance(pacmanPosition, food) for food in foodList], default=0)
    closestGhostDist = min([util.manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates], default=0)
    score = currentGameState.getScore()
    capsules = currentGameState.getCapsules()
    closestCapsuleDist = min([util.manhattanDistance(pacmanPosition, capsule) for capsule in capsules], default=0)
    closestFoodWeightedDist = min([util.manhattanDistance(pacmanPosition, food) * foodGrid[food[0]][food[1]] for food in foodList], default=0)
    closestScaredGhostDist = min([util.manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates if ghost.scaredTimer > 0], default=0)

    evaluation = score - 1.5 * closestGhostDist / (closestFoodDist + 1) + closestScaredGhostDist - closestFoodWeightedDist + 2.5 / (closestCapsuleDist + 1)
    return evaluation

better = betterEvaluationFunction
"""

def betterEvaluationFunction(currentGameState):
    
    # Evaluate the current game state to determine its value.

    
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    ghostStates = currentGameState.getGhostStates()

    # Calcular la distancia a la comida más cercana
    closestFoodDist = min([util.manhattanDistance(pacmanPosition, food) for food in foodList], default=0)

    # Calcular la distancia al fantasma más cercano
    closestGhostDist = min([util.manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates], default=0)

    # Calcular el puntaje actual del juego
    score = currentGameState.getScore()

    # Modificar el puntaje basado en las características del estado actual
    evaluation = score - 1.5 * closestGhostDist + 2 / (closestFoodDist + 1)

    return evaluation



better = betterEvaluationFunction
