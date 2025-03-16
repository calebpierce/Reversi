import numpy as np
import socket, pickle, random, multiprocessing, time, logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from reversi import reversi

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

class ReversiAgent:
    def __init__(self):
        self.board_cache = {}
        # Flag to control using parallel minimax at the root
        self.use_parallel_minimax = True

    ### UTILITY FUNCTIONS ###
    def get_legal_moves(self, board, player):
        game = reversi()
        game.board = board
        return [(i, j) for i in range(8) for j in range(8) if game.step(i, j, player, False) > 0]

    def apply_move(self, board, move, player):
        temp_board = board.copy()
        game = reversi()
        game.board = temp_board
        game.step(move[0], move[1], player, True)
        return temp_board

    def completes_edge(self, board, player):
        """Return True if any edge is completely occupied by 'player'."""
        return (np.all(board[0, :] == player) or
                np.all(board[7, :] == player) or
                np.all(board[:, 0] == player) or
                np.all(board[:, 7] == player))

    def opponent_can_complete_edge(self, board, opponent):
        """Simulate opponent moves; return True if at least one leads to full edge control."""
        legal_moves = self.get_legal_moves(board, opponent)
        for move in legal_moves:
            new_board = self.apply_move(board, move, opponent)
            if self.completes_edge(new_board, opponent):
                return True
        return False

    def evaluate_board(self, board, player):
        """
        Dynamic evaluation function with phase-based weights, a penalty for frontier discs,
        an edge control heuristic, and a new diagonal control heuristic (mid game only).
        """
        piece_count = np.count_nonzero(board)
        # Set weights based on the phase of the game
        if piece_count < 20:
            corner_value = 20
            edge_value = 5
            piece_value = 1
            mobility_value = 15
        elif piece_count < 50:
            corner_value = 25
            edge_value = 10
            piece_value = 1
            mobility_value = 10
        else:
            corner_value = 30
            edge_value = 15
            piece_value = 2
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

        # Evaluate each disc based on its location
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

        # Add mobility bonus
        mobility = len(self.get_legal_moves(board, player))
        score += mobility * mobility_value

        # --- Edge Control Heuristic for Mid Game ---
        if 20 <= piece_count < 50:
            edge_bonus = 0
            # Extract edges from the board
            top_edge = board[0, :]
            bottom_edge = board[7, :]
            left_edge = board[:, 0]
            right_edge = board[:, 7]

            def edge_control_bonus(edge):
                my_count = np.sum(edge == player)
                opp_count = np.sum(edge == -player)
                bonus = 5 * (my_count - opp_count)  # baseline partial control bonus
                if my_count == 8:
                    bonus += 50  # complete control bonus
                elif opp_count == 8:
                    bonus -= 50  # opponent full control penalty
                return bonus

            edge_bonus += edge_control_bonus(top_edge)
            edge_bonus += edge_control_bonus(bottom_edge)
            edge_bonus += edge_control_bonus(left_edge)
            edge_bonus += edge_control_bonus(right_edge)

            score += edge_bonus

            # --- Diagonal Control Heuristic for Mid Game ---
            diag_bonus = 0
            # Main diagonal (top-left to bottom-right)
            main_diag = np.array([board[i, i] for i in range(8)])
            # Anti-diagonal (top-right to bottom-left)
            anti_diag = np.array([board[i, 7-i] for i in range(8)])

            def diag_control_bonus(diag):
                my_count = np.sum(diag == player)
                opp_count = np.sum(diag == -player)
                bonus = 5 * (my_count - opp_count)  # baseline partial control bonus
                if my_count == 8:
                    bonus += 50  # complete control bonus
                elif opp_count == 8:
                    bonus -= 50  # opponent full control penalty
                return bonus

            diag_bonus += diag_control_bonus(main_diag)
            diag_bonus += diag_control_bonus(anti_diag)
            score += diag_bonus
            # --- End Diagonal Control Heuristic ---

        return score

    ### GREEDY ALGORITHM FOR EARLY GAME ###
    def greedy_move(self, board, player):
        logging.debug("Using Greedy Algorithm (Early Game)")
        best_move = (-1, -1)
        max_flips = 0
        game = reversi()
        game.board = board

        for move in self.get_legal_moves(board, player):
            flips = game.step(move[0], move[1], player, False)
            if flips > max_flips:
                max_flips = flips
                best_move = move

        return best_move

    ### MINIMAX WITH ALPHA-BETA PRUNING ###
    def minimax(self, board, depth, alpha, beta, maximizing_player, player, start_time, time_limit=4.99):
        if time.time() - start_time >= time_limit:
            logging.debug("Time limit reached in Minimax, returning default value")
            return (float('-inf') if maximizing_player else float('inf')), None

        board_tuple = tuple(map(tuple, board))
        if board_tuple in self.board_cache:
            return self.board_cache[board_tuple]

        legal_moves = self.get_legal_moves(board, player)
        if depth == 0 or not legal_moves:
            eval_score = self.evaluate_board(board, player)
            self.board_cache[board_tuple] = (eval_score, None)
            return eval_score, None

        best_move = None
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                temp_board = self.apply_move(board, move, player)
                eval_score, _ = self.minimax(temp_board, depth - 1, alpha, beta, False, -player, start_time, time_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.board_cache[board_tuple] = (max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                temp_board = self.apply_move(board, move, player)
                eval_score, _ = self.minimax(temp_board, depth - 1, alpha, beta, True, -player, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.board_cache[board_tuple] = (min_eval, best_move)
            return min_eval, best_move

    def parallel_minimax(self, board, depth, player, start_time, time_limit=4.99):
        """
        Evaluate each legal move at the root in parallel using multithreading.
        """
        legal_moves = self.get_legal_moves(board, player)
        if not legal_moves:
            return self.evaluate_board(board, player), (-1, -1)

        results = []
        # Increase parallelism: use up to 16 threads.
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_move = {
                executor.submit(self.minimax, self.apply_move(board, move, player),
                                depth - 1, float('-inf'), float('inf'), False, -player, start_time, time_limit): move
                for move in legal_moves
            }
            for future in as_completed(future_to_move):
                move = future_to_move[future]
                try:
                    score, _ = future.result()
                    results.append((score, move))
                except Exception as exc:
                    logging.error(f'Move {move} generated an exception: {exc}')

        if player == 1:
            best_score, best_move = max(results, key=lambda x: x[0])
        else:
            best_score, best_move = min(results, key=lambda x: x[0])
        return best_score, best_move

    def iterative_deepening(self, board, player, time_limit=4.9):
        start_time = time.time()
        best_move = (-1, -1)
        best_eval = float('-inf') if player == 1 else float('inf')
        depth = 1
        while time.time() - start_time < time_limit:
            current_eval, current_move = self.minimax(board, depth, float('-inf'), float('inf'), True, player, start_time, time_limit)
            if current_move is not None:
                best_move = current_move
                best_eval = current_eval
            depth += 1
        logging.debug(f"Iterative deepening completed at depth {depth-1}, best move: {best_move}")
        return best_eval, best_move

    def iterative_deepening_parallel(self, board, player, time_limit=4.9):
        """
        Iterative deepening that uses parallel minimax at the root.
        """
        start_time = time.time()
        best_move = (-1, -1)
        best_eval = float('-inf') if player == 1 else float('inf')
        depth = 1
        while time.time() - start_time < time_limit:
            current_eval, current_move = self.parallel_minimax(board, depth, player, start_time, time_limit)
            if current_move is not None:
                best_move = current_move
                best_eval = current_eval
            depth += 1
        logging.debug(f"Parallel iterative deepening completed at depth {depth-1}, best move: {best_move}")
        return best_eval, best_move

    ### NEW: Iterative Monte Carlo Simulation ###
    def monte_carlo_simulation(self, board, player, time_limit=4.99, batch_size=100):
        logging.debug("Using Monte Carlo Simulation (Late Game)")
        legal_moves = self.get_legal_moves(board, player)
        logging.debug(f"Legal moves found: {legal_moves}")
        if not legal_moves:
            logging.debug("No legal moves in Monte Carlo, passing turn.")
            return (-1, -1)

        move_scores = {move: 0 for move in legal_moves}
        start_time = time.time()
        # Increase parallelism: use 8 processes.
        pool = multiprocessing.Pool(processes=8)

        # Run batches until the remaining time is below a threshold.
        while True:
            elapsed = time.time() - start_time
            remaining_time = time_limit - elapsed
            if remaining_time < 0.1:  # if less than 0.1 seconds remain, break immediately
                break

            results = []
            for move in legal_moves:
                temp_board = self.apply_move(board, move, player)
                results.append(pool.apply_async(self.run_simulations, args=(temp_board, player, batch_size, start_time, time_limit)))
            # Try to retrieve results without exceeding the time limit.
            for idx, move in enumerate(legal_moves):
                try:
                    result_value = results[idx].get(timeout=remaining_time)
                    move_scores[move] += result_value
                except multiprocessing.TimeoutError:
                    logging.debug("Batch simulation timed out.")
            # If time is up, break out.
            if time.time() - start_time >= time_limit:
                break
        pool.terminate()

        best_move = max(move_scores, key=move_scores.get, default=(-1, -1))
        logging.debug(f"Monte Carlo selected move: {best_move} with score {move_scores.get(best_move, 0)}")
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
        logging.debug(f"Starting random game simulation for player {player}")
        game = reversi()
        game.board = board.copy()
        move_limit = 80  # Prevent infinite loops
        move_count = 0

        while move_count < move_limit:
            legal_moves = self.get_legal_moves(game.board, player)
            if not legal_moves:
                player = -player
                if not self.get_legal_moves(game.board, player):
                    break
            else:
                move = random.choice(legal_moves)
                logging.debug(f"Playing random move {move} for player {player}")
                game.step(move[0], move[1], player, True)
            player = -player
            move_count += 1

        logging.debug(f"Random game ended after {move_count} moves.")
        white_count = np.sum(game.board == 1)
        black_count = np.sum(game.board == -1)
        if white_count > black_count:
            logging.debug("White wins the random game.")
            return 1
        elif black_count > white_count:
            logging.debug("Black wins the random game.")
            return -1
        else:
            logging.debug("The random game ended in a draw.")
            return 0

    ### NEW: Safe Move Filter for Minimax ###
    def filter_safe_moves(self, board, player, candidate_moves):
        """Return only moves that do NOT allow opponent to complete an edge next turn."""
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
        self.board_cache.clear()  # Clear cache each move

        if piece_count < 20:
            best_move = self.greedy_move(board, player)
        elif piece_count < 35:
            logging.debug("Using Minimax (Mid Game)")
            if self.use_parallel_minimax:
                _, best_move = self.iterative_deepening_parallel(board, player, time_limit=4.99)
            else:
                _, best_move = self.iterative_deepening(board, player, time_limit=4.99)
            # --- Enforce Safe Move Constraint ---
            simulated_board = self.apply_move(board, best_move, player)
            if self.opponent_can_complete_edge(simulated_board, -player):
                logging.debug("Chosen minimax move allows opponent edge completion. Searching for safe alternatives...")
                candidate_moves = self.get_legal_moves(board, player)
                safe_moves = self.filter_safe_moves(board, player, candidate_moves)
                if safe_moves:
                    best_safe_move = None
                    best_eval = float('-inf') if player == 1 else float('inf')
                    for move in safe_moves:
                        temp_board = self.apply_move(board, move, player)
                        eval_score = self.evaluate_board(temp_board, player)
                        if (player == 1 and eval_score > best_eval) or (player == -1 and eval_score < best_eval):
                            best_eval = eval_score
                            best_safe_move = move
                    if best_safe_move is not None:
                        best_move = best_safe_move
            # --- End Safe Move Constraint ---
        else:
            best_move = self.monte_carlo_simulation(board, player, time_limit=4, batch_size=200)

        logging.debug(f"Best move chosen: {best_move}")
        legal_moves = self.get_legal_moves(board, player)
        if best_move not in legal_moves:
            best_move = (-1, -1)
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
                break

            legal_moves = agent.get_legal_moves(board, turn)
            if not legal_moves:
                logging.debug("No legal moves available, passing turn")
                move = (-1, -1)
            else:
                move = agent.select_best_move(board, turn)
            game_socket.send(pickle.dumps(move))
    except Exception as e:
        logging.error(f"Error during game loop: {e}")
    finally:
        game_socket.close()

if __name__ == '__main__':
    main()
