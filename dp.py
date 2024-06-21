import random
from connect4_bitboards import *
from dp_policies import *
import torch

threshold = 0.0001
gamma = 0.99
NUM_EPISODES = 1

def policy_evaluation(env: Connect4BitboardEnv, policy, gamma, threshold):
    V = {}
    max_delta = threshold + 1
    while max_delta > threshold:
        temp = V.copy()
        max_delta = 0
        for state, action in policy.items():
            env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
            env.current_player = 0  # Set to the player's turn

            if env.heights[action] < env.board_height:
                original_board = env.board.copy()
                original_heights = env.heights.copy()
                _, reward, done, _ = env.step(action)
                new_state = (env.board[0], env.board[1])
                new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
                temp[state] = reward + (gamma * V.get(new_state_idx, 0) if not done else reward)
                max_delta = max(max_delta, abs(V.get(state, 0) - temp[state]))

                # Revert the board state
                env.board = original_board
                env.heights = original_heights
        V = temp
    return V

def policy_improvement(env, V, gamma):
    policy = {}
    for state in V.keys():
        env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
        env.current_player = 0  # Set to the player's turn
        best_action_value = -float('inf')
        best_action = None
        for action in range(env.action_space.n):
            if env.heights[action] < env.board_height:
                original_board = env.board[:]
                original_heights = env.heights[:]
                _, reward, done, _ = env.step(action)
                new_state = (env.board[0], env.board[1])
                new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
                action_value = reward + (gamma * V.get(new_state_idx, 0) if not done else reward)
                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = action

                # Revert the board state
                env.board = original_board
                env.heights = original_heights
        if best_action is not None:
            policy[state] = best_action
    return policy

def policy_iteration(env, gamma=0.99, threshold=0.0001):
    policy = {}
    # Initialize policy with random actions for all possible states encountered
    initial_policy = {}
    for _ in range(NUM_EPISODES):
        state = env.reset()
        done = False
        while not done:
            state_idx = (state[0] << (env.board_height * env.board_width)) | state[1]
            if state_idx not in initial_policy:
                possible_moves = env.get_possible_moves()
                rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
                initial_policy[state_idx] = rand_action
            action = initial_policy[state_idx]
            if env.heights[action] < env.board_height:
                state, reward, done, _ = env.step(action)
    policy = initial_policy

    while True:
        V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold)
        new_policy = policy_improvement(env, V, gamma=gamma)
        if new_policy == policy:
            return V, new_policy
        policy = new_policy



env = Connect4BitboardEnv()
obs = env.reset()

# def run_episode(env, policy_to_play, opponent_policy):
#     state = obs
#     total_reward = 0
#     done = False
#     env.reset()
#     while not done:
#         current_player = env.current_player
#         if current_player == 0:  # agent's turn
#             state_idx = (state[0] << (env.board_height * env.board_width)) | state[1]
#             possible_moves = env.get_possible_moves()
#             rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]

#             action = policy_to_play.get(state_idx, rand_action)
#             while env.heights[action] >= env.board_height:  # Ensure the action is valid
#                 possible_moves = env.get_possible_moves()
#                 action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
#                 policy_to_play[state_idx] = action

#         else:  # opponent's turn
#             action_probs = opponent_policy(env, state, env.heights, current_player=current_player)
#             print(action_probs.keys())
#             # action = max(action_probs, key=action_probs.get)
#             action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
#         state, reward, done, _ = env.step(action)
#         if current_player == 0:
#             total_reward += reward
#         if done:
#             print(current_player, action, state)
#     # print(total_reward)
#     return total_reward, policy_to_play

def run_episode(env, policy_to_play, opponent_policy):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        current_player = env.current_player
        if current_player == 0:  # agent's turn
            state_idx = (int(state[0]) << (env.board_height * env.board_width)) | int(state[1])
            possible_moves = env.get_possible_moves()
            rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
            action = policy_to_play.get(state_idx, rand_action)
            # Ensure the action is valid
            policy_to_play[state_idx] = action
        else:  # opponent's turn
            action_probs = opponent_policy(env.clone(), state, env.heights, current_player)
            if sum(list(action_probs.values())) != 1:
                action = action = max(action_probs, key=action_probs.get)
            else:
                action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
        state, reward, done, _ = env.step(action)
        if current_player == 0:
            total_reward += reward
        if done:
            print(current_player, action, state)
    return total_reward, policy_to_play
while True:
    V_optimal, optimal_policy = policy_iteration(env)
    total_reward = []
    for n in range(NUM_EPISODES):
        reward, policy = run_episode(env, optimal_policy, minimax_opponent_policy)
        total_reward.append(reward)
    print(f"Success rate over {NUM_EPISODES} episodes: {sum(total_reward) * 100 / NUM_EPISODES}%")
    with open(f'optimal_policy_dp.json', 'w') as f:
        import json
        json.dump(list(zip(total_reward, policy.items())), f)
# env.render()
