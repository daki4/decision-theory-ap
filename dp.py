import copy
from connect4_bitboards import Connect4BitboardEnv
from dp_policies import random_opponent_policy, minimax_opponent_policy
import torch
import numpy as np
import json

threshold = 0.0001
gamma = 0.5
NUM_EPISODES = 10000


def policy_evaluation(env, policy, gamma, threshold, opponent_policy):
    V = {}
    max_delta = threshold + 1
    while max_delta > threshold:
        temp = V.copy()
        max_delta = 0
        for state, action in policy.items():
            env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
            env.current_player = 0  # Set to the player's turn

            if env.heights[action] < env.board_height:
                # Generate transition probabilities
                original_board = copy.deepcopy(env.board)
                original_heights = copy.deepcopy(env.heights)
                transition_probs = env.generate_transition_probabilities(env.board, lambda e, s, h, p: opponent_policy(e, s, h, p))

                value = 0
                for prob_action, (new_state, prob) in transition_probs.items():
                    if prob_action == action:
                        new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
                        reward = 1 if env._check_win(new_state[0]) else 0
                        pr = prob if isinstance(prob, (int, float)) else prob[action]
                        value += pr * (reward + gamma * V.get(new_state_idx, 0))

                temp[state] = value
                max_delta = max(max_delta, abs(V.get(state, 0) - temp[state]))

                # Revert the board state
                env.board = original_board
                env.heights = original_heights
        V = temp
    return V

def policy_improvement(env, V, gamma, opponent_policy):
    policy = {}
    for state in V.keys():
        env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
        env.current_player = 0  # Set to the player's turn
        best_action_value = -float('inf')
        best_action = None
        for action in env.get_possible_moves():
            if env.heights[action] < env.board_height:
                original_board = copy.deepcopy(env.board)
                original_heights = copy.deepcopy(env.heights)
                transition_probs = env.generate_transition_probabilities(env.board, lambda e, s, h, p: opponent_policy(e, s, h, p))

                action_value = 0
                for prob_action, (new_state, prob) in transition_probs.items():
                    if prob_action == action:
                        new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
                        reward = 1 if env._check_win(new_state[0]) else 0
                        pr = prob if isinstance(prob, (int, float)) else prob[action]
                        action_value += pr * (reward + gamma * V.get(new_state_idx, 0))

                if action_value > best_action_value:
                    best_action_value = action_value
                    best_action = action

                # Revert the board state
                env.board = original_board
                env.heights = original_heights
        if best_action is not None:
            policy[state] = best_action
    return policy

def policy_iteration(env, gamma=0.95, threshold=0.00001, opponent_policy=None):
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
        V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold, opponent_policy=opponent_policy)
        new_policy = policy_improvement(env, V, gamma=gamma, opponent_policy=opponent_policy)
        if new_policy == policy:
            return V, new_policy
        policy = new_policy

env = Connect4BitboardEnv()
obs = env.reset()

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
            action_probs = opponent_policy(env.clone(), state, env.heights, current_player, orig_interface=True)
            if sum(list(action_probs.values())) != 1:
                action = max(action_probs, key=action_probs.get)
            else:
                action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
        state, reward, done, _ = env.step(action)
        if current_player == 0:
            total_reward += reward
        if done:
            print(f'player: {current_player}, reward: {reward}, action: {action}, state: {state}')
    return total_reward, policy_to_play

# Running the Policy Iteration
g_opponent_policy = minimax_opponent_policy
overall_winrate = []
runtime = 0

max_success_rate_percent = 0.0

while True:
    runtime += 1
    env.reset()

    V_optimal, optimal_policy = policy_iteration(env, opponent_policy=random_opponent_policy)

    print("optimal policy:", optimal_policy)
    total_reward = []
    EPISODES_TO_RUN = 10
    for n in range(EPISODES_TO_RUN):
        reward, policy = run_episode(env, optimal_policy, minimax_opponent_policy)
        total_reward.append(reward)
    success_rate_percent = sum(total_reward) * 100 / EPISODES_TO_RUN
    print(f"Success rate over {EPISODES_TO_RUN} episodes: {success_rate_percent}%")

    overall_winrate.append(sum(total_reward) * 100 / EPISODES_TO_RUN)
    print(sum(overall_winrate) / runtime)
    with open(f'optimal_policy_dp.json', 'w') as f:
        import json
        json.dump(list(zip(total_reward, policy.items())), f)


# OLD CODE ####################################
# import copy
# from connect4_bitboards import *
# from dp_policies import *
# import torch

# threshold = 0.0001
# gamma = 0.5
# NUM_EPISODES = 2000

# def policy_evaluation(env: Connect4BitboardEnv, policy, gamma, threshold, opponent_policy):
#     V = {}
#     max_delta = threshold + 1
#     while max_delta > threshold:
#         temp = V.copy()
#         max_delta = 0
#         for state, action in policy.items():
#             env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
#             env.current_player = 0  # Set to the player's turn

#             if env.heights[action] < env.board_height:
#                 # Generate transition probabilities
#                 original_board = copy.deepcopy(env.board)
#                 original_heights = copy.deepcopy(env.heights)
#                 transition_probs = env.generate_transition_probabilities(env.board, lambda e, s, h, p: opponent_policy(e, s, h, p))
                
#                 value = 0
#                 for prob_action, (new_state, prob) in transition_probs.items():
#                     if prob_action == action:
#                         new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
#                         reward = 1 if env._check_win(new_state[0]) else 0
#                         if isinstance(prob, dict):
#                             if isinstance(prob[action], tuple):
#                                 pr = prob[action][1]
#                             else:
#                                 pr = prob[action]
#                         elif isinstance(prob, int):
#                             pr = prob
#                         elif isinstance(prob, float):
#                             pr = prob
#                         else:
#                             pr = prob[action]
#                         value += pr * (reward + gamma * V.get(new_state_idx, 0))
                
#                 temp[state] = value
#                 max_delta = max(max_delta, abs(V.get(state, 0) - temp[state]))

#                 # Revert the board state
#                 env.board = original_board
#                 env.heights = original_heights
#         V = temp
#     return V

# def policy_improvement(env, V, gamma, opponent_policy):
#     policy = {}
#     for state in V.keys():
#         env.board = [(state >> env.board_height * env.board_width) & ((1 << (env.board_height * env.board_width)) - 1), state & ((1 << (env.board_height * env.board_width)) - 1)]
#         env.current_player = 0  # Set to the player's turn
#         best_action_value = -float('inf')
#         best_action = None
#         for action in env.get_possible_moves():
#             if env.heights[action] < env.board_height:
#                 original_board = copy.deepcopy(env.board)
#                 original_heights = copy.deepcopy(env.heights)
#                 transition_probs = env.generate_transition_probabilities(env.board, lambda e, s, h, p: opponent_policy(e, s, h, p))
                
#                 action_value = 0
#                 for prob_action, (new_state, prob) in transition_probs.items():
#                     if prob_action == action:
#                         new_state_idx = (new_state[0] << (env.board_height * env.board_width)) | new_state[1]
#                         reward = 1 if env._check_win(new_state[0]) else 0
#                         if isinstance(prob, dict):
#                             if isinstance(prob[action], tuple):
#                                 pr = prob[action][1]
#                             else:
#                                 pr = prob[action]
#                         elif isinstance(prob, int):
#                             pr = prob
#                         elif isinstance(prob, float):
#                             pr = prob
#                         else:
#                             pr = prob[action]
#                         action_value += pr * (reward + gamma * V.get(new_state_idx, 0))
                
#                 if action_value > best_action_value:
#                     best_action_value = action_value
#                     best_action = action

#                 # Revert the board state
#                 env.board = original_board
#                 env.heights = original_heights
#         if best_action is not None:
#             policy[state] = best_action
#     return policy

# def policy_iteration(env, gamma=0.95, threshold=0.00001, opponent_policy=None):
#     policy = {}
#     # Initialize policy with random actions for all possible states encountered
#     initial_policy = {}
#     for _ in range(NUM_EPISODES):
#         state = env.reset()
#         done = False
#         while not done:
#             state_idx = (state[0] << (env.board_height * env.board_width)) | state[1]
#             if state_idx not in initial_policy:
#                 possible_moves = env.get_possible_moves()
#                 rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
#                 initial_policy[state_idx] = rand_action
#             action = initial_policy[state_idx]
#             if env.heights[action] < env.board_height:
#                 state, reward, done, _ = env.step(action)
#     policy = initial_policy

#     while True:
#         V = policy_evaluation(env, policy, gamma=gamma, threshold=threshold, opponent_policy=opponent_policy)
#         new_policy = policy_improvement(env, V, gamma=gamma, opponent_policy=opponent_policy)
#         if new_policy == policy:
#             return V, new_policy
#         policy = new_policy



# env = Connect4BitboardEnv()
# obs = env.reset()

# # def run_episode(env, policy_to_play, opponent_policy):
# #     state = obs
# #     total_reward = 0
# #     done = False
# #     env.reset()
# #     while not done:
# #         current_player = env.current_player
# #         if current_player == 0:  # agent's turn
# #             state_idx = (state[0] << (env.board_height * env.board_width)) | state[1]
# #             possible_moves = env.get_possible_moves()
# #             rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]

# #             action = policy_to_play.get(state_idx, rand_action)
# #             while env.heights[action] >= env.board_height:  # Ensure the action is valid
# #                 possible_moves = env.get_possible_moves()
# #                 action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
# #                 policy_to_play[state_idx] = action

# #         else:  # opponent's turn
# #             action_probs = opponent_policy(env, state, env.heights, current_player=current_player)
# #             print(action_probs.keys())
# #             # action = max(action_probs, key=action_probs.get)
# #             action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
# #         state, reward, done, _ = env.step(action)
# #         if current_player == 0:
# #             total_reward += reward
# #         if done:
# #             print(current_player, action, state)
# #     # print(total_reward)
# #     return total_reward, policy_to_play

# def run_episode(env, policy_to_play, opponent_policy):
#     state = env.reset()
#     total_reward = 0
#     done = False
#     while not done:
#         current_player = env.current_player
#         if current_player == 0:  # agent's turn
#             state_idx = (int(state[0]) << (env.board_height * env.board_width)) | int(state[1])
#             possible_moves = env.get_possible_moves()
#             rand_action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
#             action = policy_to_play.get(state_idx, rand_action)
#             # Ensure the action is valid
#             policy_to_play[state_idx] = action
#         else:  # opponent's turn
#             action_probs = opponent_policy(env.clone(), state, env.heights, current_player, orig_interface=True)
#             if sum(list(action_probs.values())) != 1:
#                 action = action = max(action_probs, key=action_probs.get)
#             else:
#                 action = np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))
#         state, reward, done, _ = env.step(action)
#         if current_player == 0:
#             total_reward += reward
#         if done:
#             print(f'player: {current_player}, reward: {reward}, action: {action}, state: {state}')
#     return total_reward, policy_to_play
# g_opponent_policy = minimax_opponent_policy
# overall_winrate = []
# runtime = 0

# max_success_rate_percent = 0.0

# while True:
#     runtime += 1
#     env.reset()

#     # if max_success_rate_percent < 50.0:
#     #     V_optimal, optimal_policy = policy_iteration(env, opponent_policy=random_opponent_policy)


#     V_optimal, optimal_policy = policy_iteration(env, opponent_policy=random_opponent_policy)


#     print("optimal policy:", optimal_policy)
#     total_reward = []
#     EPISODES_TO_RUN = 10
#     for n in range(EPISODES_TO_RUN):
#         reward, policy = run_episode(env, optimal_policy, minimax_opponent_policy)
#         total_reward.append(reward)
#     success_rate_percent = sum(total_reward) * 100 / EPISODES_TO_RUN
#     print(f"Success rate over {EPISODES_TO_RUN} episodes: {success_rate_percent}%")

#     # if success_rate_percent > max_success_rate_percent:
#     #     max_success_rate_percent = success_rate_percent

#     overall_winrate.append(sum(total_reward) * 100 / EPISODES_TO_RUN)
#     print(sum(overall_winrate) / runtime)
#     with open(f'optimal_policy_dp.json', 'w') as f:
#         import json
#         json.dump(list(zip(total_reward, policy.items())), f)
# # env.render()
