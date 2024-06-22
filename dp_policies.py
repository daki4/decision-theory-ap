from functools import lru_cache
import numpy as np

from connect4_bitboards import Connect4BitboardEnv
from concurrent.futures import ProcessPoolExecutor, as_completed


# Example opponent policy: Random move
def random_opponent_policy(env: Connect4BitboardEnv, state, heights, current_player, *args, **kwargs):
    valid_actions = [col for col in range(env.board_width) if heights[col] < env.board_height]
    if not valid_actions:
        return 0.0

    # Uniform random policy for the opponent
    action_probs = {action: 1 / len(valid_actions) for action in valid_actions}
    return action_probs


def bitboard_to_list(bitboard, board_height, board_width):
    board_list = [[0 for _ in range(board_width)] for _ in range(board_height)]
    for col in range(board_width):
        for row in range(board_height):
            pos = col * (board_height + 1) + row
            if (bitboard >> pos) & 1:
                board_list[row][col] = 1
    return board_list

def evaluate_window(window, piece):
    score = 0
    opp_piece = 1 if piece == 2 else 2
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(0) == 1:
        score -= 4
    return score

def score_position(bitboard, piece, board_height, board_width):
    score = 0
    board_list = bitboard_to_list(bitboard, board_height, board_width)
    opp_piece = 1 if piece == 2 else 2

    # Score center column
    center_array = [board_list[r][board_width//2] for r in range(board_height)]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(board_height):
        row_array = board_list[r]
        for c in range(board_width - 3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(board_width):
        col_array = [board_list[r][c] for r in range(board_height)]
        for r in range(board_height - 3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(board_height - 3):
        for c in range(board_width - 3):
            window = [board_list[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(board_height - 3):
        for c in range(board_width - 3):
            window = [board_list[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)

    return score

def cache(user_function, /):
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    return lru_cache(maxsize=8192)(user_function)

@cache
def evaluate_board(env: Connect4BitboardEnv, state, current_player):
    if env._check_win(state[0]):
        return 10000000 if current_player == 0 else -100000
    if env._check_win(state[1]):
        return 10000000 if current_player == 1 else -100000
    piece = 1 if current_player == 1 else 2
    opp_piece = 1 if piece == 2 else 2
    score = score_position(state[current_player], piece, env.board_height, env.board_width)
    score -= score_position(state[1 - current_player], opp_piece, env.board_height, env.board_width)
    return score

def minimax(env, depth, alpha=-np.inf, beta=np.inf, maximizing_player=False):
    current_player = env.current_player
    state = env._get_obs()

    if (
        depth == 0
        or env._check_win(state[0])
        or env._check_win(state[1])
        or all(h >= env.board_height for h in env.heights)
    ):
        return evaluate_board(env, state, current_player)

    if maximizing_player:
        max_eval = -np.inf
        for action in env.get_possible_moves():
            new_env = env.clone()
            new_env.step(action)

            eval = minimax(new_env, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)

            if alpha >= beta:
                break
        return max_eval
    else:
        min_eval = np.inf
        for action in env.get_possible_moves():
            new_env = env.clone()
            new_env.step(action)

            eval = minimax(new_env, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval

def evaluate_action(action, env, depth, maximizing_player):
    new_env = env.clone()
    try:
        new_env.step(action)
    except ValueError:
        return action, -np.inf if maximizing_player else np.inf

    move_value = minimax(new_env, depth - 1, alpha=-np.inf, beta=np.inf, maximizing_player=not maximizing_player)
    return action, move_value

def minimax_opponent_policy(envv, state, heights, current_player, depth=6, orig_interface=False):
    best_action = None
    env = envv.clone()
    best_value = -np.inf if env.current_player == current_player else np.inf  # Adjust for player 1
    maximizing_player = (env.current_player == current_player)
    weights = {}
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_action, action, env, depth, maximizing_player) for action in env.get_possible_moves()]
        for future in as_completed(futures):
            action, move_value = future.result()
            weights[action] = move_value
            if (maximizing_player and move_value > best_value) or (not maximizing_player and move_value < best_value):
                best_value = move_value
                best_action = action

            # Debugging statement
            # print(f"Action: {action}, Move Value: {move_value}, Best Value: {best_value}, Best Action: {best_action}")

    # print(f"Move Taken: {action}")
    if not orig_interface:
        action_probs = {
            action: (0.0, weights.get(action, -1000000000000)) for action in range(env.board_width)
        }  # Ensure range matches the board width
    else:
        # print(weights)
        action_probs = {
            action: 0.0 for action in range(env.board_width)
        }  # Ensure range matches the board width

    if best_action is not None:
        action_probs[best_action] = 1.0

    return action_probs


def play_human_vs_minimax():
    env = Connect4BitboardEnv()
    state = env.reset()
    env.render()

    while True:
        if env.current_player == 0:  # Human player
            valid_move = False
            while not valid_move:
                try:
                    human_action = int(input("Enter your move (0-6): "))
                    if human_action not in env.get_possible_moves():
                        raise ValueError
                    env.step(human_action)
                    valid_move = True
                except ValueError:
                    print("Invalid move. Try again.")
        else:  # Minimax player
            print("Minimax is thinking...")
            minimax_action = minimax_opponent_policy(env, env.board, env.heights, env.current_player, depth=6, orig_interface=True)
            action = max(minimax_action, key=minimax_action.get)
            env.step(action)

        env.render()

        if env._check_win(env.board[0]):
            print("Player 0 (Human) wins!")
            break
        if env._check_win(env.board[1]):
            print("Player 1 (Minimax) wins!")
            break
        if all(h == env.board_height for h in env.heights):
            print("It's a draw!")
            break

# Start the game
if __name__ == '__main__':
    play_human_vs_minimax()
