from collections import defaultdict
from multiprocessing import Pool, cpu_count
import pickle
from connect4_bitboards import *
import numpy as np
import torch

# class MCTSNode:
#     def __init__(self, state, parent=None, action=None):
#         self.state = state
#         self.parent = parent
#         self.action = action
#         self.children = []
#         self.wins = 0
#         self.visits = 0
#         self.untried_actions = parent.get_possible_moves() if parent else env.get_possible_moves()
    
#     def is_fully_expanded(self):
#         return len(self.untried_actions) == 0

#     def best_child(self, c_param=1.4):
#         choices_weights = [(child.wins / child.visits) + c_param * np.sqrt((2 * np.log(self.visits) / child.visits)) for child in self.children]
#         return self.children[np.argmax(choices_weights)]

# def select(node):
#     while node.is_fully_expanded() and node.children:
#         node = node.best_child()
#     return node

# def expand(node):
#     action = node.untried_actions.pop()
#     env.board = node.state
#     next_state, _, _, _ = env.step(action)
#     child_node = MCTSNode(state=next_state, parent=node, action=action)
#     node.children.append(child_node)
#     return child_node

# def simulate(state):
#     env.board = state
#     env.current_player = 0
#     done = False
#     while not done:
#         possible_moves = env.get_possible_moves()
#         action = possible_moves[torch.randint(len(possible_moves), (1,)).item()]
#         state, reward, done, _ = env.step(action)
#     return reward

# def backpropagate(node, reward):
#     while node:
#         node.visits += 1
#         node.wins += reward
#         node = node.parent

# def mcts(root, n_iter=1000):
#     for _ in range(n_iter):
#         node = select(root)
#         if not node.is_fully_expanded():
#             node = expand(node)
#         reward = simulate(node.state)
#         backpropagate(node, reward)
#     return root.best_child(c_param=0).action

# env = Connect4BitboardEnv()
# obs = env.reset()
# root = MCTSNode(state=obs)

# best_action = mcts(root, n_iter=1000)
# print(f"Best action: {best_action}")


import copy
import random
import math
import numpy as np

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_moves())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def most_visited_child(self):
        visits = [child.visits for child in self.children]
        return self.children[np.argmax(visits)]

    def expand(self):
        moves = self.state.get_possible_moves()
        for move in moves:
            if not any(child.move == move for child in self.children):
                next_state = copy.deepcopy(self.state)
                next_state.step(move)
                child_node = Node(next_state, self, move)
                self.children.append(child_node)
                return child_node

    def rollout(self):
        current_state = copy.deepcopy(self.state)
        while not current_state._check_win(current_state.board[0]) and not current_state._check_win(current_state.board[1]) and current_state.get_possible_moves():
            possible_moves = current_state.get_possible_moves()
            move = random.choice(possible_moves)
            current_state.step(move)
        if current_state._check_win(current_state.board[0]):
            return 1 if current_state.current_player == 1 else -1
        elif current_state._check_win(current_state.board[1]):
            return 1 if current_state.current_player == 0 else -1
        else:
            return 0

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)

class MCTS:
    def __init__(self, env, q_table, num_simulations=1000):
        self.env = env
        self.q_table = q_table
        self.num_simulations = num_simulations

    def search(self, state):
        root = Node(state)

        for _ in range(self.num_simulations):
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            if not node.is_fully_expanded():
                node = node.expand()

            result = node.rollout()
            node.backpropagate(result)
            self.update_q_table(node)

        return root.best_child().move

    def update_q_table(self, node):
        while node:
            key = (node.state._get_obs(), node.move)
            if key not in self.q_table:
                self.q_table[key] = 0
            self.q_table[key] += node.value
            node = node.parent

def merge_q_tables(q_table_list):
    merged_q_table = defaultdict(lambda: 0)
    for q_table in q_table_list:
        for key, value in q_table.items():
            merged_q_table[key] += value
    return dict(merged_q_table)

import random
from collections import defaultdict

# Initialize the global episode counter
episode = 0

def get_state_action_key(state, action):
    state_str = f'({state[0]}, {str(state[1])})'
    return (state_str, action)

def run_episode(args):
    global episode
    episode += 1
    env, num_simulations = args
    # env = Connect4BitboardEnv()
    q_table = defaultdict(lambda: 0)
    mcts = MCTS(env, q_table, num_simulations=num_simulations)

    state = env.reset()
    done = False
    while not done:
        if env.current_player == 0:
            action = mcts.search(env)
        else:
            possible_moves = env.get_possible_moves()
            action = random.choice(possible_moves)

        next_state, reward, done, _ = env.step(action)

        key = get_state_action_key(state, action)
        q_table[key] += reward

        state = next_state

        if done:
            break
    print(f'episode {episode} done')

    return dict(q_table)


def train_mcts(env, num_episodes=160, num_simulations=1000):
    with Pool(processes=cpu_count()) as pool:
        q_tables = pool.map(run_episode, [(env, num_simulations) for _ in range(num_episodes)])

    merged_q_table = merge_q_tables(q_tables)

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(merged_q_table, f)

    with open('qtable.json', 'w') as f:
       import json
       json.dump({str(k): v for k, v in merged_q_table.items()}, f, indent=4)

    print("Q-table generated and saved to 'q_table.pkl'")
env = Connect4BitboardEnv()
rewards = train_mcts(env, num_episodes=160, num_simulations=1000)
print(f"Average reward over 100 episodes: {np.mean(rewards)}")
