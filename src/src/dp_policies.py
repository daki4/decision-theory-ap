
import numpy as np

from connect4_bitboards import Connect4BitboardEnv

# Example opponent policy: Random move
def random_opponent_policy(env, state, heights, current_player, *args, **kwargs):
    valid_actions = [col for col in range(7) if heights[col] < 6]
    if not valid_actions:
        return 0.0
    
    # Uniform random policy for the opponent
    action_probs = {action: 1 / len(valid_actions) for action in valid_actions}
    return action_probs

def minimax(env, state, heights, current_player, depth, alpha=-np.inf, beta=np.inf, maximizing_player=True):
    env.board = state
    env.heights = heights.copy()
    env.current_player = current_player

    if depth == 0 or env._check_win(state[0]) or env._check_win(state[1]):
        if env._check_win(state[0]):
            return 1 if maximizing_player else -1
        if env._check_win(state[1]):
            return -1 if maximizing_player else 1
        return 0  # Draw or depth limit reached

    if maximizing_player:
        max_eval = -np.inf
        for action in env.get_possible_moves():
            if heights[action] >= env.board_height:  # Check for full columns
                continue

            new_board = list(state)
            new_heights = heights.copy()
            move = 1 << (action * (env.board_height + 1) + new_heights[action])
            new_board[current_player] ^= move
            new_heights[action] += 1

            eval = minimax(env, new_board, new_heights, 1 - current_player, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)

            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for action in env.get_possible_moves():
            if heights[action] >= env.board_height:  # Check for full columns
                continue

            new_board = list(state)
            new_heights = heights.copy()
            move = 1 << (action * (env.board_height + 1) + new_heights[action])
            new_board[current_player] ^= move
            new_heights[action] += 1

            eval = minimax(env, new_board, new_heights, 1 - current_player, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break
        
        return min_eval

def minimax_opponent_policy(env, state, heights, current_player, depth=10):
    best_action = None
    best_value = -np.inf if current_player == 0 else np.inf
    new_board = env

    for action in env.get_possible_moves():
        if heights[action] >= env.board_height:  # Check for full columns
            continue

        new_board = list(state)
        new_heights = heights.copy()
        move = 1 << (action * (env.board_height + 1) + new_heights[action])
        new_board[current_player] ^= move
        new_heights[action] += 1

        move_value = minimax(env, new_board, new_heights, 1 - current_player, depth - 1, alpha=-np.inf, beta=np.inf, maximizing_player=(current_player == 0))

        if (current_player == 0 and move_value > best_value) or (current_player == 1 and move_value < best_value):
            best_value = move_value
            best_action = action

    action_probs = {action: 0.0 for action in range(env.board_width)}  # Ensure range matches the board width
    if best_action is not None:
        action_probs[best_action] = 1.0

    return action_probs
