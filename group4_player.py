import numpy as np
import socket, pickle
from reversi import reversi
import os
import pandas as pd

PRIORITY_MATRIX = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 3, 2, 2, 3, -2, 10],
    [5, -2, 2, 1, 1, 2, -2, 5],
    [5, -2, 2, 1, 1, 2, -2, 5],
    [10, -2, 3, 2, 2, 3, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100]
]


def evaluateBoard(board):
    max_score = 0
    min_score = 0
    for i in range(8):
        for j in range(8):
            if board[i][j] == 1:  # White (Maximizing)
                max_score += PRIORITY_MATRIX[i][j]
            elif board[i][j] == -1:  # Black (Minimizing)
                min_score += PRIORITY_MATRIX[i][j]
    return max_score - min_score

def calculate_score(board):
    white_score = np.sum(board == 1)
    black_score = np.sum(board == -1)

    return white_score, black_score

def isGameComplete(depth):
    return depth == 4

def has_valid_moves(game, player):
    for i in range(8):
        for j in range(8):
            if game.step(i, j, player, False) > 0:  # Check if the move is valid
                return True
    return False

def min_max(game, depth, isMaxPlayer, alpha, beta):
    if isGameComplete(depth):
        return evaluateBoard(game.board)
    if not has_valid_moves(game, 1 if isMaxPlayer else -1):
        return evaluateBoard(game.board)

    if isMaxPlayer:
        max_score = -float('inf')
        for i in range(8):
            for j in range(8):
                if game.step(i, j, 1, False) > 0:  # Check valid move
                    temp_game = reversi()
                    temp_game.board = np.copy(game.board)  # Copy game state
                    temp_game.step(i, j, 1, True)  # Apply move

                    score = min_max(temp_game, depth + 1, False, alpha, beta)
                    max_score = max(max_score, score)
                    alpha = max(alpha, score)

                    if beta <= alpha:  # Pruning condition
                        return max_score
        return max_score
    else:
        min_score = float('inf')
        for i in range(8):
            for j in range(8):
                if game.step(i, j, -1, False) > 0:  # Check valid move
                    temp_game = reversi()
                    temp_game.board = np.copy(game.board)  # Copy game state
                    temp_game.step(i, j, -1, True)  # Apply move

                    score = min_max(temp_game, depth + 1, True, alpha, beta)
                    min_score = min(min_score, score)
                    beta = min(beta, score)

                    if beta <= alpha:  # Pruning condition
                        return min_score
        return min_score


def player_turn(game, turn):
    best_score = -float('inf') if turn == 1 else float('inf')  # Adjust based on player type
    x = y = -1

    for i in range(8):
        for j in range(8):
            if game.step(i, j, turn, False) > 0:  # Check valid move
                temp_game = reversi()
                temp_game.board = np.copy(game.board)  # Copy game state
                temp_game.step(i, j, turn, True)  # Apply move

                score = min_max(temp_game, 0, turn == -1, -float('inf'), float('inf'))

                if (turn == 1 and score > best_score) or (turn == -1 and score < best_score):
                    best_score = score
                    x, y = i, j

    return x, y

def save_score(isWhite, white_score, black_score):
    filename = "scores.csv"

    if isWhite:
        df = pd.DataFrame([(white_score, black_score)], columns=["Our AI", "Greedy AI"])
    else:
        df = pd.DataFrame([(black_score, white_score)], columns=["Our AI", "Greedy AI"])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index= False)


def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        try:
            data = game_socket.recv(4096)
            if not data:
                break  # Exit if connection is closed

            turn, board = pickle.loads(data)

            print(turn)

            if turn == 1:
                isWhite = True
                print("Our AI is white")
            elif turn == -1:
                isWhite= False
                print("Our AI is black")
            elif turn == 0:
                white_score, black_score = calculate_score(board)
                print("Game Over")
                print(f"Score: White-{white_score} , Black-{black_score}")
                # save_score(isWhite, white_score, black_score)
                break

            #print(f"Turn: {turn}")
            #print(np.rot90(board))

            game.board = board  # Update the game state
            x, y = player_turn(game, turn)

            game_socket.send(pickle.dumps([x, y]))

        except Exception as e:
            #print(f"Error: {e}")
            break

    game_socket.close()


if __name__ == "__main__":
    main()
