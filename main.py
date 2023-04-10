from __future__ import division

from collections import defaultdict
from multiprocessing import Process, Lock

import numpy as np
import random
from copy import deepcopy

# MCTSPY
from mctspy.tree.nodes import MonteCarloTreeSearchNode
from mctspy.games.examples.tictactoe import TicTacToeGameState

# Graphs
import seaborn as sns
sns.set(color_codes=True)

class TicTacToeState():
    def __init__(self, board=None, first_player=1):
        if board is not None:
            if len(board.shape) != 2 or (board.shape[0] != board.shape[1]):
                raise ValueError("Only 2D square boards allowed")
            else:
                self.board = np.array(board)
        else:
            self.board = np.zeros((3,3))

        self.board_size = self.board.shape[0]

        if first_player not in [1,-1]:
            raise ValueError("First player must be equal to 1 (player) or -1 (computer).")
        #self.currentPlayer = first_player
        self.next_to_move = first_player

        self.corners = [[0,0],
                        [0, self.board_size-1],
                        [self.board_size-1, 0],
                        [self.board_size-1, self.board_size-1]]
        self.center = [int(self.board_size/2), int(self.board_size/2)]


    def get_legal_actions(self):
        """Given a current state, return all the posssible actions"""
        return [Action(player=self.next_to_move, x=idx[0], y=idx[1])
                for idx in np.argwhere(self.board==0)]


    def move(self, action):
        newState = deepcopy(self)
        newState.board[action.x][action.y] = action.player
        newState.next_to_move = self.next_to_move * -1
        return newState


    def takeRandomAction(self):
        return self.move(action=random.choice(self.get_legal_actions()))


    def takeSmartAction(self):
        """ Play a 'smart' action:
            - win if it can
            - block opponent if it can win next
            - otherwise, play a action following the rule :
                first the corners then the center, otherwise a random available position """
        possible_actions = self.get_legal_actions()

        # win if it can
        for action in possible_actions:
            newState = self.move(action)
            if newState.gameResult() == [True, action.player]:
                return self.move(action)

        # block opponent if is going to win
        for action in possible_actions:
            actionOpponent = deepcopy(action)
            actionOpponent.player = action.player * -1
            newState = self.move(actionOpponent)
            if newState.gameResult() == [True, actionOpponent.player]:
                return self.move(action)

        # play randomly
        return self.move(action=random.choice(possible_actions))


    def gameResult(self):
        for row in self.board:
            if abs(sum(row)) == self.board_size:
                is_terminal = True
                reward = sum(row) / self.board_size
                return [is_terminal, reward]

        for column in self.board.T:
            if abs(sum(column)) == self.board_size:
                is_terminal = True
                reward = sum(column) / self.board_size
                return [is_terminal, reward]

        first_diag = self.board.trace()
        second_diag = self.board[::-1].trace()

        for diagonal in [first_diag, second_diag]:
            if abs(diagonal) == self.board_size:
                is_terminal = True
                reward = diagonal / self.board_size
                return [is_terminal, reward]

        if np.all(self.board != 0):
            is_terminal = True
            reward = 0
            return [is_terminal, reward]

        return [0, None]


    def is_game_over(self):
        return self.gameResult()[0]


    def game_result(self):
        return self.gameResult()[1]


    def __repr__(self):
        """print game board"""
        res = '---------\n'
        for row in self.board:
            for idx, val in enumerate(row):
                end = ' | '
                if idx == len(row)-1: end = ' \n'
                res += self.getMarker(val) + end
            res += '---------\n'

        return res


    @staticmethod
    def getMarker(val):
        """Transform an integer (1, -1 or 0) into a marker ('X', 'O' or ' ')."""
        if val == 1:
            return 'X'
        elif val == -1:
            return 'O'
        elif val == 0:
            return ' '
        else:
            print('Value must be equal to 1, -1 or O.')
        return None


class Action():
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))

class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)


class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number=1000, c_param=1.4):
        """
        Parameters
        ----------
        simulations_number: int
            number of simulations performed to get the best action
        c_param: double
            exploration constant
        Returns
        -------
        """
        for _ in range(simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        # to select best child go for exploitation only
        return self.root.best_child(c_param=c_param)


    def _tree_policy(self):
        """
        selects node to run rollout/playout for
        Returns
        -------
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
def playTicTacToe(board=None,
                  method_player='mcts',
                  method_computer='random',
                  first_player=1,
                  print_board=False,
                  return_board=False,
                  simulations_number=1000,
                  c_param=1.0):
    """
    Play a game of Tic Tac Toe automatically between a 'player' and the computer.

    Parameters:
        board: 2D array, the board to play Tic Tac Toc,
            if None, initialize with np.zeros((3,3))
        method_player: string, method to play against the computer
            default 'mcts' (see Methods for other values)
        method_computer: string, method used by the computer to play
            default 'random' (see Methods for other values)
        first_player: integer, default 1
            (1: the "user" plays first, -1: the computer plays first)
        simulations_number: integer, the number of iterations for a MCTS.
        c_param: double, the exploration constant for the MCTS

    Methods:
        random: play a random move among those possible
        smart: win if can, block its opponent if he can win next, or moves randomly
        mcts: select the best action to do using Monte Carlo tree search
    """

    game = TicTacToeState(board=board, first_player=first_player)

    list_board = []

    while not game.is_game_over():
        if game.next_to_move == 1:
            method = method_player
        elif game.next_to_move == -1:
            method = method_computer
        else:
            raise ValueError("Current player must be equal to 1 (player) or -1 (computer).")

        if method == 'random':
            game = game.takeRandomAction()
        elif method == 'smart':
            game = game.takeSmartAction()
        elif method == 'mcts':
            initial_board_state = TicTacToeGameState(state=game.board, next_to_move=game.next_to_move)
            root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state)
            best_node = MonteCarloTreeSearch(root).best_action(simulations_number=simulations_number, c_param=c_param)
            game.board = best_node.state.board
            game.next_to_move *= -1
        else:
            raise ValueError("Method unknown. Please select a method in ['random', 'smart', 'mcts'].")

        if print_board: print(game)
        if return_board: list_board.append(game.board)

    if return_board:
        return game.game_result(), list_board
    else:
        return game.game_result()

if __name__ == "__main__":
    playTicTacToe(board=None,
                  method_player='mcts',
                  method_computer='smart',
                  first_player=-1, # 1: player first, -1: computer first
                  print_board=True,
                  simulations_number=1000,
                  c_param=0)
