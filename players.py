import random
import time
from copy import deepcopy
import numpy as np

import pygame
import math


class connect4Player(object):
    def __init__(self, position, seed=0):
        self.position = position
        self.opponent = None
        self.seed = seed
        random.seed(seed)

    def play(self, env, move):
        move = [-1]


class human(connect4Player):

    def play(self, env, move):
        move[:] = [int(input('Select next move: '))]
        while True:
            if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
                break
            move[:] = [int(input('Index invalid. Select next move: '))]


class human2(connect4Player):

    def play(self, env, move):
        done = False
        while (not done):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if self.position == 1:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                    else:
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))
                    move[:] = [col]
                    done = True


class randomAI(connect4Player):
    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        move[:] = [random.choice(indices)]


class stupidAI(connect4Player):
    def play(self, env, move):
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        if 3 in indices:
            move[:] = [3]
        elif 2 in indices:
            move[:] = [2]
        elif 1 in indices:
            move[:] = [1]
        elif 5 in indices:
            move[:] = [5]
        elif 6 in indices:
            move[:] = [6]
        else:
            move[:] = [0]


class minimaxAI(connect4Player):

    def drop_coin(self, env, col, isMax):
        row = env.topPosition[col] + 1
        env.board[row][col] = 0
        if isMax:
            env.history[0].pop()
        else:
            env.history[1].pop()
        env.topPosition[col] = env.topPosition[col] + 1

    # def eval_function(self, env):

        # p1 = 0
        # p2 = 0
        # for i in range(6):
        #     for j in range(5):
        #         if env.board[i][j] == env.board[i][j+1] == env.board[i][j+2] and env.board[i][j] == 1:
        #             p1 = p1 + 1
        #         elif env.board[i][j] == env.board[i][j+1] == env.board[i][j+2] and env.board[i][j] == 2:
        #             p2 = p2 + 1
        #
        # for i in range(4):
        #     for j in range(6):
        #         if env.board[i][j] == env.board[i+1][j] == env.board[i+2][j] and env.board[i][j] == 1:
        #             p1 = p1 + 1
        #         elif env.board[i][j] == env.board[i+1][j] == env.board[i+2][j] and env.board[i][j] == 2:
        #             p2 = p2 + 1
        #
        # for i in range(4):
        #     for j in range(5):
        #         if env.board[i][j] == env.board[i+1][j+1] == env.board[i+2][j+2] and env.board[i][j] == 1:
        #             p1 = p1 + 1
        #         elif env.board[i][j] == env.board[i+1][j+1] == env.board[i+2][j+2] and env.board[i][j] == 2:
        #             p2 = p2 + 1
        #
        # for i in range(4):
        #     for j in range(5):
        #         if env.board[i+2][j] == env.board[i+1][j+1] == env.board[i][j+2] and env.board[i][j] == 1:
        #             p1 = p1 + 1
        #         elif env.board[i+2][j] == env.board[i+1][j+1] == env.board[i][j+2] and env.board[i][j] == 2:
        #             p2 = p2 + 1

        # result = p1 * 2 - p2
        # return result


    def eval_function(self, env):

        weights = [[9, 3, 2, 1, 2, 3, 6],
                   [9, 3, 2, 1, 2, 3, 6],
                   [9, 3, 2, 1, 2, 3, 6],
                   [9, 4, 2, 1, 2, 4, 6],
                   [9, 5, 5, 1, 2, 5, 6],
                   [9, 5, 1, 1, 1, 5, 6]]

        env.getBoard()
        weight_matrix = np.array(weights, np.int32) * -1
        b = np.where(env.getBoard() == 2, 1, 0)
        a = np.where(env.getBoard() == 1, 1, 0)
        result = np.sum(a * weight_matrix)*5 - np.sum(b * weight_matrix)
        return result

    def possible_indices(self, board):
        board_shape = board.shape
        top_positions = (np.ones(board_shape[1]) * (board_shape[0] - 1)).astype('int32')
        possible = top_positions >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        return indices

    def minimax(self, env, depth, isMax):
        env = deepcopy(env)
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)
        if isMax:
            last_player = 2
        else:
            last_player = 1

        column = env.history[last_player - 1][-1]
        if depth == 0 or env.gameOver(column, 1) or env.gameOver(column, 2):
            if env.gameOver(column, last_player) and not isMax:
                value = -1 * math.inf #* (depth + 1)
                return (value, column)
            elif env.gameOver(column, last_player) and isMax:
                value = math.inf #* (depth + 1)
                return (value, column)
            else:
                return (self.eval_function(env), column)

        if not isMax:
            maxEval = -1 * math.inf
            cur_max_col = random.choice(indices)
            for i in indices:
                self.simulateMove(env, i, 1)
                eval = self.minimax(env, depth - 1, False)[0]
                self.drop_coin(env, env.history[0][-1], True)
                if eval > maxEval:
                    cur_max_col = i
                maxEval = max(maxEval, eval)
            return (maxEval, cur_max_col)

        else:
            minEval = math.inf
            cur_min_col = random.choice(indices)
            for i in indices:
                self.simulateMove(env, i, 2)
                eval = self.minimax(env, depth - 1, True)[0]
                self.drop_coin(env, env.history[1][-1], False)
                if eval < minEval:
                    cur_min_col = i
                minEval = min(minEval, eval)
            return (minEval, cur_min_col)

    def play(self, env, move):
        player = env.turnPlayer.position  # get whose turn it is
        isMax = player == 1  # if max player, how will this tell you if you are first or second
        depth = 3

        if np.sum(env.board) == 0:  # first move
            move[:] = [3]
        else:
            col = [self.minimax(env, depth, isMax)[1]]
            move[:] = col

    def simulateMove(self, env, move, player):
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[player - 1].append(move)


class alphaBetaAI(connect4Player):

    def drop_coin(self, env, col, isMax):
        row = env.topPosition[col] + 1
        env.board[row][col] = 0
        if isMax:
            env.history[0].pop()
        else:
            env.history[1].pop()
        env.topPosition[col] = env.topPosition[col] + 1

    # def eval_function(self, env):
    #     p1 = 0
    #     p2 = 0
    #     for i in range(6):
    #         for j in range(5):
    #             if env.board[i][j] == env.board[i][j+1] == env.board[i][j+2] and env.board[i][j] == 1:
    #                 p1 = p1 + 1
    #             elif env.board[i][j] == env.board[i][j+1] == env.board[i][j+2] and env.board[i][j] == 2:
    #                 p2 = p2 + 1
    #
    #     for i in range(4):
    #         for j in range(6):
    #             if env.board[i][j] == env.board[i+1][j] == env.board[i+2][j] and env.board[i][j] == 1:
    #                 p1 = p1 + 1
    #             elif env.board[i][j] == env.board[i+1][j] == env.board[i+2][j] and env.board[i][j] == 2:
    #                 p2 = p2 + 1
    #
    #     for i in range(4):
    #         for j in range(5):
    #             if env.board[i][j] == env.board[i+1][j+1] == env.board[i+2][j+2] and env.board[i][j] == 1:
    #                 p1 = p1 + 1
    #             elif env.board[i][j] == env.board[i+1][j+1] == env.board[i+2][j+2] and env.board[i][j] == 2:
    #                 p2 = p2 + 1
    #
    #     for i in range(4):
    #         for j in range(5):
    #             if env.board[i+2][j] == env.board[i+1][j+1] == env.board[i][j+2] and env.board[i][j] == 1:
    #                 p1 = p1 + 1
    #             elif env.board[i+2][j] == env.board[i+1][j+1] == env.board[i][j+2] and env.board[i][j] == 2:
    #                 p2 = p2 + 1
    #
    #     result = p1 - p2
    #     return result

    def eval_function(self, env):

        weights = [[2, 5, 1, 6, 1, 5, 2],
                   [2, 5, 2, 6, 2, 5, 2],
                   [1, 6, 1, 1, 1, 6, 1],
                   [1, 5, 2, 1, 2, 5, 1],
                   [3, 2, 1, 1, 1, 2, 3],
                   [1, 2, 1, 1, 1, 2, 1]]
        env.getBoard()
        weight_matrix = np.array(weights, np.int32) * -1
        b = np.where(env.getBoard() == 2, 1, 0)
        a = np.where(env.getBoard() == 1, 1, 0)
        result = np.sum(a * weight_matrix) * 3 - np.sum(b * weight_matrix)
        return result

    def sorted_from_middle(self, lst, reverse=False):
        left = lst[len(lst) // 2 - 1::-1]
        right = lst[len(lst) // 2:]
        output = [right.pop(0)] if len(lst) % 2 else []
        for t in zip(left, right):
            output += sorted(t, reverse=reverse)
        return output

    def minimax(self, env, depth, isMax, alpha, beta):
        env = deepcopy(env)
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: indices.append(i)

        indices = self.sorted_from_middle(indices)

        if isMax:
            last_player = 2
        else:
            last_player = 1

        column = env.history[last_player - 1][-1]
        if depth == 0 or env.gameOver(column, 1) or env.gameOver(column, 2):
            if env.gameOver(column, last_player) and not isMax:
                value = -1 * (depth + 1) * 100000
                return (value, column)
            elif env.gameOver(column, last_player) and isMax:
                value = (depth + 1) * 100000
                return (value, column)
            else:
                return (self.eval_function(env) * (depth + 1), column)

        if isMax:  # if max, how to check if you are the max player
            maxEval = -1 * math.inf
            cur_max_col = 3
            for i in indices:
                self.simulateMove(env, i, 1)
                eval = self.minimax(env, depth - 1, False, alpha, beta)[0]
                self.drop_coin(env, env.history[0][-1], True)
                if eval > maxEval:
                    cur_max_col = i
                maxEval = max(maxEval, eval)
                alpha = max(alpha, maxEval)
                if alpha >= beta:
                    print("did sumn")
                    break
            return (maxEval, cur_max_col)

        else:
            minEval = math.inf
            cur_min_col = 3
            for i in indices:
                self.simulateMove(env, i, 2)
                eval = self.minimax(env, depth - 1, True, alpha, beta)[0]
                self.drop_coin(env, env.history[1][-1], False)
                if eval < minEval:
                    cur_min_col = i
                minEval = min(minEval, eval)
                beta = min(beta, minEval)
                if alpha >= beta:
                    print("did sumn 22")
                    break
            return (minEval, cur_min_col)

    def play(self, env, move):
        player = env.turnPlayer.position  # get whose turn it is
        isMax = player == 1  # if max player, how will this tell you if you are first or second
        depth = 2

        if np.sum(env.board) == 0:  # first move
            move[:] = [3]
            # print("staring move as column 3")
        else:
            alpha = -1 *math.inf
            beta =  math.inf
            col = [self.minimax(env, depth, isMax, alpha, beta)[1]]
            move[:] = col

    def simulateMove(self, env, move, player):
        env.board[env.topPosition[move]][move] = player
        env.topPosition[move] -= 1
        env.history[player - 1].append(move)


SQUARESIZE = 100
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)
