import logging
import multiprocessing
import pickle
import random
import socket
import time

import numpy as np

from reversi import reversi

# ===================== PARAMETERS =====================
DEBUG_LOG = True  # Enable/disable debug logging
MINIMAX_TIME_LIMIT = 5.0  # Time limit (in seconds) for minimax iterative deepening
MCTS_TIME_LIMIT = 4.99  # Time limit (in seconds) for Monte Carlo simulation
MC_PROCESSES = 1  # Number of processes for Monte Carlo simulation (set to 1 to disable parallelism)
MC_BATCH_SIZE = 1000  # Batch size for Monte Carlo simulation iterations
MAX_MINIMAX_DEPTH = 4  # Maximum search depth for minimax
# Evaluation tuning constants
LOCK_ROW_EDGE_BONUS = 100
LOCK_ROW_INNER_BONUS = 50
LOCK_COL_EDGE_BONUS = 100
LOCK_COL_INNER_BONUS = 50
LOCK_DIAG_BONUS = 100
RISK_ROW_PENALTY = 20
RISK_COL_PENALTY = 20
RISK_DIAG_PENALTY = 15
# =======================================================

# Configure logging based on DEBUG_LOG flag.
if DEBUG_LOG:
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')


class ReversiAgent:
    def __init__(self):
        self.board_cache = {}
        self.use_parallel_minimax = False

    def calculate_score(self, board):
        white_score = np.sum(board == 1)
        black_score = np.sum(board == -1)

        return white_score, black_score

    ### UTILITY FUNCTIONS ###
    def get_legal_moves(self, board, player):
        game = reversi()
        game.board = board
        return [(i, j) for i in range(8) for j in range(8)
                if game.step(i, j, player, False) > 0]

    def apply_move(self, board, move, player):
        temp_board = board.copy()
        game = reversi()
        game.board = temp_board
        game.step(move[0], move[1], player, True)
        return temp_board

    # --- Minimax Helper Functions ---
    def get_valid_moves(self, board, turn):
        moves = []
        for i in range(8):
            for j in range(8):
                sim_game = reversi()
                sim_game.board = board.copy()
                gain = sim_game.step(i, j, turn, False)
                if gain > 0:
                    moves.append((i, j))
        return moves

    def simulate_move(self, board, i, j, turn):
        sim_game = reversi()
        sim_game.board = board.copy()
        sim_game.step(i, j, turn, True)
        return sim_game.board.copy()

    # --- Advanced Evaluation Function ---
    def evaluate_board(self, board, player):
        """
        Advanced evaluation function with phase-based weights, mobility, frontier penalties,
        and additional bonuses for locking down entire rows, columns, and diagonals.
        Also subtracts a risk penalty if the opponent is one move away from locking a line.
        """
        piece_count = np.count_nonzero(board)
        if piece_count < 45:
            corner_value = 20;
            edge_value = 5;
            piece_value = 1;
            mobility_value = 15
        else:
            corner_value = 30;
            edge_value = 15;
            piece_value = 2;
            mobility_value = 5

        score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(i, j) for i in [0, 7] for j in range(8)] + [(i, j) for j in [0, 7] for i in range(8)]
        edges = list(set(edges) - set(corners))

        def is_frontier(board, i, j):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 8 and 0 <= nj < 8:
                        if board[ni][nj] == 0:
                            return True
            return False

        frontier_penalty = 2
        for i in range(8):
            for j in range(8):
                if board[i][j] == player:
                    if (i, j) in corners:
                        score += corner_value
                    elif (i, j) in edges:
                        score += edge_value
                    else:
                        score += piece_value
                    if is_frontier(board, i, j):
                        score -= frontier_penalty

        mobility = len(self.get_legal_moves(board, player))
        score += mobility * mobility_value

        # Bonuses for locking down rows:
        for i in range(8):
            if np.all(board[i, :] == player):
                if i == 0 or i == 7:
                    score += LOCK_ROW_EDGE_BONUS
                else:
                    score += LOCK_ROW_INNER_BONUS

        # Bonuses for locking down columns:
        for j in range(8):
            if np.all(board[:, j] == player):
                if j == 0 or j == 7:
                    score += LOCK_COL_EDGE_BONUS
                else:
                    score += LOCK_COL_INNER_BONUS

        # Bonuses for locking down diagonals:
        main_diag = np.array([board[i, i] for i in range(8)])
        if np.all(main_diag == player):
            score += LOCK_DIAG_BONUS
        anti_diag = np.array([board[i, 7 - i] for i in range(8)])
        if np.all(anti_diag == player):
            score += LOCK_DIAG_BONUS

        # Risk penalties: subtract if opponent is one move away from locking a row, column, or diagonal.
        opponent = -player
        risk_penalty = 0
        for i in range(8):
            row = board[i, :]
            if np.count_nonzero(row == opponent) == 7 and np.count_nonzero(row == 0) == 1:
                risk_penalty += RISK_ROW_PENALTY
        for j in range(8):
            col = board[:, j]
            if np.count_nonzero(col == opponent) == 7 and np.count_nonzero(col == 0) == 1:
                risk_penalty += RISK_COL_PENALTY
        if np.count_nonzero(main_diag == opponent) == 7 and np.count_nonzero(main_diag == 0) == 1:
            risk_penalty += RISK_DIAG_PENALTY
        if np.count_nonzero(anti_diag == opponent) == 7 and np.count_nonzero(anti_diag == 0) == 1:
            risk_penalty += RISK_DIAG_PENALTY

        score -= risk_penalty

        return score

    # --- Simple Evaluation Fallback (Not used in minimax anymore) ---
    def evaluate(self, board, root_turn):
        return root_turn * np.sum(board)

    # --- End Advanced Evaluation ---

    ### DYNAMIC INITIAL DEPTH ###
    def dynamic_initial_depth(self, board, player):
        moves = self.get_legal_moves(board, player)
        count = len(moves)
        if count <= 2:
            return 2
        elif count <= 4:
            return 3
        else:
            return 4

    ### SIMPLE MINIMAX WITH TIME CONTROL ###
    def minimax_simple(self, board, turn, depth, alpha, beta, root_turn, start_time, time_limit):
        if time.time() - start_time >= time_limit:
            return self.evaluate_board(board, root_turn), (-1, -1)
        valid_moves = self.get_valid_moves(board, turn)
        if depth == 0 or (not valid_moves and not self.get_valid_moves(board, -turn)):
            return self.evaluate_board(board, root_turn), (-1, -1)
        if not valid_moves:
            return self.minimax_simple(board, -turn, depth, alpha, beta, root_turn, start_time, time_limit)
        best_move = (-1, -1)
        if turn == root_turn:
            value = -float('inf')
            for move in valid_moves:
                new_board = self.simulate_move(board, move[0], move[1], turn)
                score, _ = self.minimax_simple(new_board, -turn, depth - 1, alpha, beta, root_turn, start_time,
                                               time_limit)
                if score > value:
                    value = score
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move
        else:
            value = float('inf')
            for move in valid_moves:
                new_board = self.simulate_move(board, move[0], move[1], turn)
                score, _ = self.minimax_simple(new_board, -turn, depth - 1, alpha, beta, root_turn, start_time,
                                               time_limit)
                if score < value:
                    value = score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def iterative_deepening_time(self, board, player, time_limit=MINIMAX_TIME_LIMIT):
        start_time = time.time()
        best_move = (-1, -1)
        best_score = None
        depth = self.dynamic_initial_depth(board, player)
        while time.time() - start_time < time_limit and depth <= MAX_MINIMAX_DEPTH:
            score, move = self.minimax_simple(board, player, depth, -float('inf'), float('inf'), player, start_time,
                                              time_limit)
            if move != (-1, -1):
                best_move = move
                best_score = score
            depth += 1
        print("[DEBUG] Minimax depth reached:", depth - 1)
        return best_score, best_move

    ### ITERATIVE MONTE CARLO SIMULATION ###
    def monte_carlo_simulation(self, board, player, time_limit=MCTS_TIME_LIMIT, batch_size=MC_BATCH_SIZE):
        logging.debug("Using Monte Carlo Simulation (Late Game)")
        legal_moves = self.get_legal_moves(board, player)
        logging.debug(f"Legal moves found: {legal_moves}")
        if not legal_moves:
            logging.debug("No legal moves in Monte Carlo, passing turn.")
            return (-1, -1)
        move_scores = {move: 0 for move in legal_moves}
        total_simulations = 0
        start_time = time.time()
        pool = multiprocessing.Pool(processes=MC_PROCESSES)
        while True:
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed
            if remaining_time < 0.1:
                break
            results = []
            for move in legal_moves:
                temp_board = self.apply_move(board, move, player)
                results.append(pool.apply_async(self.run_simulations,
                                                args=(temp_board, player, batch_size, start_time, time_limit)))
            for idx, move in enumerate(legal_moves):
                try:
                    result_value = results[idx].get(timeout=remaining_time)
                    move_scores[move] += result_value
                    total_simulations += result_value
                except multiprocessing.TimeoutError:
                    logging.debug("Batch simulation timed out.")
            if time.time() - start_time >= time_limit:
                break
        pool.terminate()
        logging.debug(
            f"Total simulation iterations: {total_simulations} (Average per legal move: {total_simulations / len(legal_moves):.2f})")
        best_move = max(move_scores, key=move_scores.get, default=(-1, -1))
        logging.debug(f"Monte Carlo selected move: {best_move} with score {move_scores.get(best_move, 0)}")
        new_board = self.apply_move(board, best_move, player)
        eval_val = self.evaluate_board(new_board, player)
        predicted = 1 if eval_val >= 0 else -1
        predicted_winner = "White" if predicted == 1 else "Black"
        logging.debug(f"Predicted winner based on MCTS: {predicted_winner}")
        return best_move

    def run_simulations(self, board, player, sim_count, start_time, time_limit):
        win_count = 0
        for i in range(sim_count):
            if time.time() - start_time >= time_limit:
                break
            winner = self.simulate_random_game(board, player)
            if winner == player:
                win_count += 1
            elif winner == 0:
                win_count += 0.5
        return win_count

    def simulate_random_game(self, board, player):
        game = reversi()
        game.board = board.copy()
        move_limit = 80
        move_count = 0
        while move_count < move_limit:
            legal_moves = self.get_legal_moves(game.board, player)
            if not legal_moves:
                player = -player
                if not self.get_legal_moves(game.board, player):
                    break
            else:
                move = random.choice(legal_moves)
                game.step(move[0], move[1], player, True)
            player = -player
            move_count += 1
        white_count = np.sum(game.board == 1)
        black_count = np.sum(game.board == -1)
        if white_count > black_count:
            return 1
        elif black_count > white_count:
            return -1
        else:
            return 0

    ### Safe Move Filter (unused) ###
    def filter_safe_moves(self, board, player, candidate_moves):
        safe_moves = []
        for move in candidate_moves:
            temp_board = self.apply_move(board, move, player)
            if not self.opponent_can_complete_edge(temp_board, -player):
                safe_moves.append(move)
        return safe_moves

    ### PHASE-BASED STRATEGY SELECTION ###
    def select_best_move(self, board, player):
        start_time = time.time()
        piece_count = np.count_nonzero(board)
        self.board_cache.clear()
        if piece_count < 58:
            logging.debug("Using Minimax (Mid Game) with time limit")
            _, best_move = self.iterative_deepening_time(board, player, time_limit=MINIMAX_TIME_LIMIT)
        else:
            best_move = self.monte_carlo_simulation(board, player, time_limit=MCTS_TIME_LIMIT, batch_size=MC_BATCH_SIZE)
        logging.debug(f"Best move chosen: {best_move}")
        legal_moves = self.get_legal_moves(board, player)
        if best_move not in legal_moves:
            best_move = max(legal_moves, key=lambda m: self.evaluate_board(self.apply_move(board, m, player), player))
        return best_move


### MAIN FUNCTION ###
def main():
    agent = ReversiAgent()
    try:
        game_socket = socket.socket()
        game_socket.connect(('127.0.0.1', 33333))
    except Exception as e:
        logging.error(f"Socket connection error: {e}")
        return
    game = reversi()
    try:
        while True:
            data = game_socket.recv(4096)
            if not data:
                break
            turn, board = pickle.loads(data)
            if turn == 0:
                logging.info("Game over. Closing connection.")
                white_score, black_score = agent.calculate_score(board)
                logging.info(f"Score: White-{white_score} , Black-{black_score}")
                break
            legal_moves = agent.get_legal_moves(board, turn)
            if not legal_moves:
                logging.debug("No legal moves available, passing turn")
                move = (-1, -1)
            else:
                move = agent.select_best_move(board, turn)
            game_socket.send(pickle.dumps(move))
            # At the end of the turn, print the group color.
            if turn == 1:
                logging.debug("Group4 is White")
            elif turn == -1:
                logging.debug("Group4 is Black")
    except Exception as e:
        logging.error(f"Error during game loop: {e}")
    finally:
        game_socket.close()


if __name__ == '__main__':
    main()
