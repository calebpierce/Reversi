#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi

# Search depth for the minimax algorithm; adjust as needed.
DEPTH = 3

def evaluate(board, root_turn):
    return root_turn * np.sum(board)

def get_valid_moves(board, turn):
    moves = []
    for i in range(8):
        for j in range(8):
            sim_game = reversi()
            sim_game.board = board.copy()
            # Check move without updating the board
            gain = sim_game.step(i, j, turn, False)
            if gain > 0:
                moves.append((i, j))
    return moves

def simulate_move(board, i, j, turn):
    sim_game = reversi()
    sim_game.board = board.copy()
    sim_game.step(i, j, turn, True)
    return sim_game.board.copy()

def minimax(board, turn, depth, alpha, beta, root_turn):
    valid_moves = get_valid_moves(board, turn)
    
    
    if depth == 0 or (not valid_moves and not get_valid_moves(board, -turn)):
        return evaluate(board, root_turn), (-1, -1)
    

    if not valid_moves:
        return minimax(board, -turn, depth, alpha, beta, root_turn)
    
    best_move = (-1, -1)
    
    if turn == root_turn:  # Maximizing player
        value = -float('inf')
        for move in valid_moves:
            new_board = simulate_move(board, move[0], move[1], turn)
            score, _ = minimax(new_board, -turn, depth - 1, alpha, beta, root_turn)
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move
    else:  # Minimizing player
        value = float('inf')
        for move in valid_moves:
            new_board = simulate_move(board, move[0], move[1], turn)
            score, _ = minimax(new_board, -turn, depth - 1, alpha, beta, root_turn)
            if score < value:
                value = score
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        # Receive play request from the server.
        # turn: 1 for white, -1 for black; board: 8x8 numpy array.
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended.
        if turn == 0:
            game_socket.close()
            return
        
        # Debug 
        print("Turn:", turn)
        print("Board:")
        print(board)

        # Apply minimax algorithm 
        _, best_move = minimax(board, turn, DEPTH, -float('inf'), float('inf'), turn)
        x, y = best_move
        
        # Send your move to the server.
        # (x, y) = (-1, -1) indicates that no legal move exists.
        game_socket.send(pickle.dumps([x, y]))
        
if __name__ == '__main__':
    main()
