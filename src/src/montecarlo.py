
# import numpy as np
from environment_connect4 import Connect4Env

# def monte_carlo_simulation(env: Connect4Env, num_simulations=1000):
#     win_counts = np.zeros((env.num_players, env.action_space.spaces[0].n))
    
#     for _ in range(num_simulations):
#         for player in range(env.num_players):
#             for action in range(env.action_space.spaces[player].n):
#                 state = env.reset()
#                 current_action = action
#                 done = False
#                 reward = 0
#                 while not done:
#                     state, reward, done, _ = env.step(current_action)
#                     if not done:
#                         current_action = np.random.choice(env.get_moves())
#                 if reward[player] == 1:
#                     win_counts[player][action] += 1
 
#     win_probabilities = win_counts / num_simulations
#     return win_probabilities



# class TDLearningAgent:
#     def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
#         self.env = env
#         self.alpha = alpha  # Learning rate
#         self.gamma = gamma  # Discount factor
#         self.epsilon = epsilon  # Exploration rate
#         self.value_table = {}  # State value table

#     def get_state_value(self, state):
#         state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
#         if state_tuple not in self.value_table:
#             self.value_table[state_tuple] = 0
#         return self.value_table[state_tuple]

#     def update_state_value(self, state, player, reward, next_state):
#         state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
#         next_state_value = self.get_state_value(next_state)
#         current_state_value = self.get_state_value(state)
#         self.value_table[state_tuple] += self.alpha * (reward[player] + self.gamma * next_state_value - current_state_value)
#         # self.value_table[state_tuple] += self.alpha * (reward[not player] + self.gamma * next_state_value - current_state_value)
    
#     def choose_action(self):
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.env.get_moves())
#         else:
#             values = []
#             for action in self.env.get_moves():
#                 next_state, _, _, _ = self.env.clone().step(action)
#                 values.append(self.get_state_value(next_state))
#             return self.env.get_moves()[np.argmax(values)]

#     def train(self, episodes=1000):
#         for episode in range(episodes):
#             state = self.env.reset()
#             done = False
#             turn = 1
#             while not done:
#                 action = self.choose_action()
#                 next_state, reward, done, _ = self.env.step(action)
#                 self.update_state_value(state, turn % 2, reward, next_state)
#                 state = next_state
#                 turn += 1
#             if (episode + 1) % 100 == 0:
#                 print(f"Episode {episode + 1}/{episodes} completed.")

# # Run the simulation
# # env = Connect4Env()
# #win_probabilities = monte_carlo_simulation(env)
# #print("Win probabilities:", win_probabilities)


# # agent = TDLearningAgent(env)
# # agent.train(episodes=1000)
# # with open('qtable.out', 'w') as f:
# #     import json
# #     json.dump({str(k): v for k, v in agent.value_table.items()}, f, indent=4)





import numpy as np
import multiprocessing as mp

# class TDLearningAgent:
#     def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=0.5, min_epsilon=0.1, epsilon_decay=0.995):
#         self.env = env
#         self.alpha = alpha  # Learning rate
#         self.gamma = gamma  # Discount factor
#         self.epsilon = epsilon  # Exploration rate
#         self.epsilon_decay = epsilon_decay # decay rate
#         self.min_epsilon = min_epsilon
#         self.value_table = {}  # State value table

#     def get_state_value(self, state):
#         state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
#         if state_tuple not in self.value_table:
#             self.value_table[state_tuple] = 0
#         return self.value_table[state_tuple]

#     def update_state_value(self, state, reward, next_state):
#         state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
#         next_state_value = self.get_state_value(next_state)
#         current_state_value = self.get_state_value(state)
#         self.value_table[state_tuple] += self.alpha * (reward + self.gamma * next_state_value - current_state_value)

#     def choose_action(self):
        
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.env.get_moves())
#         else:
#             values = []
#             for action in self.env.get_moves():
#                 next_state, _, _, _ = self.env.clone().step(action)
#                 values.append(self.get_state_value(next_state))
#             return self.env.get_moves()[np.argmax(values)]

#     def train(self, episodes):
#         for episode in range(episodes):
#             state = self.env.reset()
#             done = False
#             while not done:
#                 action = self.choose_action()
#                 next_state, reward, done, _ = self.env.step(action)
#                 self.update_state_value(state, reward[self.env.current_player - 1], next_state)
#                 state = next_state
#                 # play for the other player
#                 self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
#                 if not done:
#                     enemy = self.choose_action()
#                     _, reward, done, _ = self.env.step(enemy)
#                     self.update_state_value(state, -reward[self.env.current_player - 1], next_state)
#             if (episode + 1) % 100 == 0:
#                 print(f"Episode {episode + 1}/{episodes} completed.")

# def train_agent(episodes):
#     env = Connect4Env()
#     agent = TDLearningAgent(env)
#     agent.train(episodes)
#     return agent.value_table

# def merge_value_tables(value_tables):
#     merged_table = {}
#     for table in value_tables:
#         for key, value in table.items():
#             if key in merged_table:
#                 merged_table[key] += value
#             else:
#                 merged_table[key] = value
#     return merged_table

# if __name__ == '__main__':
#     num_processes = mp.cpu_count()
#     episodes_per_process = 16_000 // num_processes

#     with mp.Pool(num_processes) as pool:
#         value_tables = pool.map(train_agent, [episodes_per_process] * num_processes)

#     merged_value_table = merge_value_tables(value_tables)
    
#     with open('qtable.out', 'w') as f:
#         import json
#         json.dump({str(k): v for k, v in merged_value_table.items()}, f, indent=4)











# import numpy as np
# import random
# import math
# import copy
# from multiprocessing import Pool, cpu_count, Manager

# class Node:
#     def __init__(self, state, parent=None, move=None):
#         self.state = state
#         self.parent = parent
#         self.move = move
#         self.children = []
#         self.visits = 0
#         self.value = 0

#     def is_fully_expanded(self):
#         return len(self.children) == len(self.state.get_moves())

#     def best_child(self, c_param=1.4):
#         choices_weights = [
#             (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
#             for child in self.children
#         ]
#         return self.children[np.argmax(choices_weights)]

#     def most_visited_child(self):
#         visits = [child.visits for child in self.children]
#         return self.children[np.argmax(visits)]

#     def expand(self):
#         moves = self.state.get_moves()
#         for move in moves:
#             if not any(child.move == move for child in self.children):
#                 next_state = copy.deepcopy(self.state)
#                 next_state.step(move)
#                 child_node = Node(next_state, self, move)
#                 self.children.append(child_node)
#                 return child_node

#     def rollout(self):
#         current_state = copy.deepcopy(self.state)
#         while current_state.winner is None:
#             possible_moves = current_state.get_moves()
#             move = random.choice(possible_moves)
#             current_state.step(move)
#         return current_state.get_result(1 - current_state.current_player)

#     def backpropagate(self, result):
#         self.visits += 1
#         self.value += result
#         if self.parent:
#             self.parent.backpropagate(-result)


# def parallel_rollout(node_state, width, height, connect):
#     state = copy.deepcopy(node_state)
#     current_player = state.current_player
#     winner = None

#     def get_moves(state):
#         moves = []
#         for col in range(width):
#             if state[col][height - 1] == -1:
#                 moves.append(col)
#         return moves

#     def does_move_win(state, x, y, player):
#         for dx, dy in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
#             p = 1
#             while is_on_board(x + p * dx, y + p * dy) and state[x + p * dx][y + p * dy] == player:
#                 p += 1
#             n = 1
#             while is_on_board(x - n * dx, y - n * dy) and state[x - n * dx][y - n * dy] == player:
#                 n += 1
#             if p + n >= (connect + 1):
#                 return True
#         return False

#     def is_on_board(x, y):
#         return 0 <= x < width and 0 <= y < height

#     while winner is None:
#         possible_moves = get_moves(state.board)
#         move = random.choice(possible_moves)
#         row = height - 1
#         while row >= 0 and state.board[move][row] == -1:
#             row -= 1
#         row += 1
#         state.board[move][row] = current_player
#         if does_move_win(state.board, move, row, current_player):
#             winner = current_player
#         elif get_moves(state.board) == []:
#             winner = -1
#         current_player = 1 - current_player

#     if winner == -1:
#         return 0
#     return +1 if winner == state.current_player else -1


# class MCTS:
#     def __init__(self, env, q_table, num_simulations=1000):
#         self.env = env
#         self.q_table = q_table
#         self.num_simulations = num_simulations

#     def search(self, state):
#         root = Node(state)

#         for _ in range(self.num_simulations):
#             node = root
#             while node.is_fully_expanded() and node.children:
#                 node = node.best_child()

#             if not node.is_fully_expanded():
#                 node = node.expand()

#             result = parallel_rollout(node.state, self.env.width, self.env.height, self.env.connect)
#             node.backpropagate(result)
#             self.update_q_table(node)

#         return root.most_visited_child().move

#     def update_q_table(self, node):
#         while node:
#             state_str = ''.join(map(str, node.state.board.flatten()))
#             key = (state_str, node.move)
#             if key not in self.q_table:
#                 self.q_table[key] = 0
#             self.q_table[key] += node.value
#             node = node.parent

# def get_state_action_key(state, action):
#     state_str = ''.join(map(str, state.flatten()))
#     return (state_str, action)

# def get_state_action_key(state, action):
#     state_str = ''.join(map(str, state.flatten()))
#     return (state_str, action)


# def run_episode(args):
#     q_table, num_simulations = args
#     env = Connect4Env()
#     mcts = MCTS(env, q_table, num_simulations=num_simulations)

#     state = env.reset()
#     done = False
#     while not done:
#         state_str = ''.join(map(str, state.flatten()))
#         if env.current_player == 0:
#             action = mcts.search(env.clone())
#         else:
#             possible_moves = env.get_moves()
#             action = random.choice(possible_moves)

#         next_state, reward, done, info = env.step(action)

#         key = get_state_action_key(state, action)
#         if key not in q_table:
#             q_table[key] = 0

#         q_table[key] += reward[env.current_player]
#         state = next_state

#         if done:
#             break

#     return q_table


# if __name__ == "__main__":
#     num_episodes = 100
#     num_simulations = 10

#     with Manager() as manager:
#         q_table = manager.dict()

#         with Pool(processes=cpu_count()) as pool:
#             q_tables = pool.map(run_episode, [(q_table, num_simulations) for _ in range(num_episodes)])

#         q_table = dict(q_table)  # Convert manager.dict() to a regular dictionary

#     with open('qtable.out', 'w') as f:
#         import json
#         json.dump({str(k): v for k, v in q_table.items()}, f, indent=4)

#     print("Q-table generated and saved to 'q_table.pkl'")



import numpy as np
import random
import math
import copy
import pickle
from multiprocessing import Pool, cpu_count

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_moves())

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
        moves = self.state.get_moves()
        for move in moves:
            if not any(child.move == move for child in self.children):
                next_state = copy.deepcopy(self.state)
                next_state.step(move)
                child_node = Node(next_state, self, move)
                self.children.append(child_node)
                return child_node

    def rollout(self):
        current_state = copy.deepcopy(self.state)
        while current_state.winner is None:
            possible_moves = current_state.get_moves()
            move = random.choice(possible_moves)
            current_state.step(move)
        return current_state.get_result(1 - current_state.current_player)

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
        # return root.most_visited_child().move

    def update_q_table(self, node):
        while node:
            #state_str = ''.join(map(str, node.state.board.flatten()))
            #key = (state_str, node.move)
            key = get_state_action_key(node.state.get_player_observations(), node.move)
            if key not in self.q_table:
                self.q_table[key] = 0
            self.q_table[key] += node.value
            node = node.parent

episode = 0
def get_state_action_key(state, action):
    state_str = '|'.join(map(str, state.flatten()))
    return (state_str, action)

# mutexed dictionary view, bad code:
#def run_episode(args):
#    global episode
#    episode += 1
#    q_table, num_simulations = args
#    env = Connect4Env()
#    mcts = MCTS(env, q_table, num_simulations=num_simulations)
#
#    state = env.reset()
#    env.step(3)
#    done = False
#    while not done:
#        if env.current_player == 0:
#            action = mcts.search(env.clone())
#        else:
#            possible_moves = env.get_moves()
#            action = random.choice(possible_moves)
#
#        next_state, reward, done, info = env.step(action)
#
#        key = get_state_action_key(state, action)
#        if key not in q_table:
#            q_table[key] = 0
#
#        q_table[key] += reward[env.current_player]
#        state = next_state
#
#        if done:
#            break
#    print(f'episode {episode} done')
#
#    return q_table
#
#def train_mcts(num_episodes=100, num_simulations=1000):
#    with Manager() as manager:
#        q_table = manager.dict()
#
#        with Pool(processes=cpu_count()) as pool:
#            pool.map(run_episode, [(q_table, num_simulations) for _ in range(num_episodes)])
#
#        q_table = dict(q_table)  # Convert manager.dict() to a regular dictionary
#
#        with open('q_table.pkl', 'wb') as f:
#            pickle.dump(q_table, f)
#
#    print("Q-table generated and saved to 'q_table.pkl'")


import random
import math
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import pickle

def merge_q_tables(q_table_list):
    merged_q_table = defaultdict(lambda: 0)
    for q_table in q_table_list:
        for key, value in q_table.items():
            merged_q_table[key] += value
    return dict(merged_q_table)

def run_episode(args):
    global episode
    episode += 1
    num_simulations = args
    env = Connect4Env()
    q_table = defaultdict(lambda: 0)
    mcts = MCTS(env, q_table, num_simulations=num_simulations)

    state = env.reset()
    env.step(3)
    done = False
    while not done:
        if env.current_player == 0:
            action = mcts.search(env.clone())
        else:
            possible_moves = env.get_moves()
            action = random.choice(possible_moves)

        next_state, reward, done, _ = env.step(action)

        key = get_state_action_key(state, action)
        q_table[key] += reward[env.current_player]
        state = next_state

        if done:
            break
    print(f'episode {episode} done')

    return dict(q_table)

def train_mcts(num_episodes=160, num_simulations=1000):
    with Pool(processes=cpu_count()) as pool:
        q_tables = pool.map(run_episode, [num_simulations for _ in range(num_episodes)])

    merged_q_table = merge_q_tables(q_tables)

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(merged_q_table, f)

    print("Q-table generated and saved to 'q_table.pkl'")


## minimax heuristic

# def winning_move(board, piece):
#     for c in range(board.shape[1] - 3):
#         for r in range(board.shape[0]):
#             if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
#                 return True

#     for c in range(board.shape[1]):
#         for r in range(board.shape[0] - 3):
#             if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
#                 return True

#     for c in range(board.shape[1] - 3):
#         for r in range(board.shape[0] - 3):
#             if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
#                 return True

#     for c in range(board.shape[1] - 3):
#         for r in range(3, board.shape[0]):
#             if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
#                 return True
#     return False

# def evaluate_window(window, piece):
#     score = 0
#     opp_piece = PLAYER_PIECE
#     if piece == PLAYER_PIECE:
#         opp_piece = AI_PIECE

#     if window.count(piece) == 4:
#         score += 100
#     elif window.count(piece) == 3 and window.count(EMPTY) == 1:
#         score += 5
#     elif window.count(piece) == 2 and window.count(EMPTY) == 2:
#         score += 2

#     if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
#         score -= 4

#     return score

# def score_position(board, piece):
#     score = 0

#     center_array = [int(i) for i in list(board[:, board.shape[1]//2])]
#     center_count = center_array.count(piece)
#     score += center_count * 3

#     for r in range(board.shape[0]):
#         row_array = [int(i) for i in list(board[r, :])]
#         for c in range(board.shape[1] - 3):
#             window = row_array[c:c+4]
#             score += evaluate_window(window, piece)

#     for c in range(board.shape[1]):
#         col_array = [int(i) for i in list(board[:, c])]
#         for r in range(board.shape[0] - 3):
#             window = col_array[r:r+4]
#             score += evaluate_window(window, piece)

#     for r in range(board.shape[0] - 3):
#         for c in range(board.shape[1] - 3):
#             window = [board[r+i][c+i] for i in range(4)]
#             score += evaluate_window(window, piece)

#     for r in range(board.shape[0] - 3):
#         for c in range(board.shape[1] - 3):
#             window = [board[r+3-i][c+i] for i in range(4)]
#             score += evaluate_window(window, piece)

#     return score

# def is_terminal_node(board):
#     return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(env)) == 0

# def minimax(env, depth, alpha, beta, maximizingPlayer):
#     valid_locations = env.get_moves()
#     is_terminal = is_terminal_node(env)
#     if depth == 0 or is_terminal:
#         if is_terminal:
#             if winning_move(board, AI_PIECE):
#                 return (None, 100000000000000)
#             elif winning_move(board, PLAYER_PIECE):
#                 return (None, -10000000000000)
#             else:
#                 return (None, 0)
#         else:
#             return (None, score_position(board, AI_PIECE))
#     if maximizingPlayer:
#         value = -math.inf
#         column = random.choice(valid_locations)
#         for col in valid_locations:
#             row = get_next_open_row(env.board, col)
#             b_copy = board.copy()
#             b_copy[row][col] = AI_PIECE
#             new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
#             if new_score > value:
#                 value = new_score
#                 column = col
#             alpha = max(alpha, value)
#             if alpha >= beta:
#                 break
#         return column, value

#     else:
#         value = math.inf
#         column = random.choice(valid_locations)
#         for col in valid_locations:
#             row = get_next_open_row(env.board, col)
#             b_copy = board.copy()
#             b_copy[row][col] = PLAYER_PIECE
#             new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
#             if new_score < value:
#                 value = new_score
#                 column = col
#             beta = min(beta, value)
#             if alpha >= beta:
#                 break
#         return column, value

# def get_next_open_row(board, col):
#     for r in range(board.shape[0]):
#         if board[r][col] == -1:
#             return r
#     return None

# def pick_best_move(env, piece):
#     valid_locations = env.get_moves()
#     best_score = -10000
#     best_col = random.choice(valid_locations)
#     for col in valid_locations:
#         row = get_next_open_row(env.board, col)
#         temp_board = env.clone()
#         temp_board[row][col] = piece
#         score = score_position(temp_board, piece)
#         if score > best_score:
#             best_score = score
#             best_col = col
#     return best_col


## minimax no heuristic
# import math
# import multiprocessing as mp

# def minimax(node, depth, alpha, beta, maximizingPlayer):
#     if depth == 0 or node.winner is not None:
#         return node.get_result(maximizingPlayer)
    
#     valid_moves = node.get_moves()
#     if maximizingPlayer:
#         value = -math.inf
#         for move in valid_moves:
#             child = node.clone()
#             child.step(move)
#             value = max(value, minimax(child, depth-1, alpha, beta, False))
#             alpha = max(alpha, value)
#             if alpha >= beta:
#                 break
#         return value
#     else:
#         value = math.inf
#         for move in valid_moves:
#             child = node.clone()
#             child.step(move)
#             value = min(value, minimax(child, depth-1, alpha, beta, True))
#             beta = min(beta, value)
#             if alpha >= beta:
#                 break
#         return value

# def minimax_worker(child, depth, alpha, beta, maximizingPlayer, return_dict, idx):
#     return_dict[idx] = minimax(child, depth, alpha, beta, maximizingPlayer)

# def minimax_decision(env, depth=4):
#     manager = mp.Manager()
#     return_dict = manager.dict()
#     jobs = []
#     best_value = -math.inf
#     best_move = None

#     valid_moves = env.get_moves()
#     for idx, move in enumerate(valid_moves):
#         child = env.clone()
#         child.step(move)
#         p = mp.Process(target=minimax_worker, args=(child, depth-1, -math.inf, math.inf, False, return_dict, idx))
#         jobs.append(p)
#         p.start()

#     for job in jobs:
#         job.join()

#     for idx, move in enumerate(valid_moves):
#         value = return_dict[idx]
#         if value > best_value:
#             best_value = value
#             best_move = move

#     return best_move


# def minimax_vs_mcts(q_table, minimax_depth=4):
#     env = Connect4Env()
#     mcts = MCTS(env, q_table, num_simulations=1000)

#     env.reset()
#     done = False

#     print("Starting a game of Connect 4!")
#     env.render()

#     while not done:
#         if env.current_player == 0:
#             print("MCTS Agent's turn:")
#             action = mcts.search(env.clone())
#         else:
#             print("Minimax Agent's turn:")
#             action = minimax_decision(env.clone(), minimax_depth)

#         _, _, done, _ = env.step(action)
#         env.render()

#         if done:
#             if env.winner == -1:
#                 print("It's a draw!")
#             elif env.winner == 0:
#                 print("MCTS Agent wins!")
#             else:
#                 print("Minimax Agent wins!")
#             break

def mcts_vs_mcts(q_table1, q_table2):
    env = Connect4Env()

    mcts1 = MCTS(env, q_table1, num_simulations=1000)
    mcts2 = MCTS(env, q_table2, num_simulations=1000)
    env.reset()
    done = False

    print("Starting a game of Connect 4!")
    env.render()

    while not done:
        if env.current_player == 0:
            print("MCTS1 Agent's turn:")
            action = mcts1.search(env.clone())
        else:
            print("MCTS2 Agent's turn:")
            action = mcts2.search(env.clone())

        _, _, done, _ = env.step(action)
        env.render()

        if done:
            if env.winner == -1:
                print("It's a draw!")
            elif env.winner == 0:
                print("MCTS1 Agent wins!")
            else:
                print("MCTS2 Agent wins!")
            break



### Step 2: Playing Phase

def human_vs_mcts(q_table):
    env = Connect4Env()
    mcts = MCTS(env, q_table, num_simulations=1000)

    env.reset()
    done = False

    print("Starting a game of Connect 4!")
    env.render()

    while not done:
        if env.current_player == 0:
            print("MCTS Agent's turn:")
            action = mcts.search(env.clone())
        else:
            valid_move = False
            while not valid_move:
                try:
                    action = int(input("Your turn! Enter the column number (0-6): "))
                    if action in env.get_moves():
                        valid_move = True
                    else:
                        print("Invalid move. Column is full or out of range. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 6.")

        _, _, done, _ = env.step(action)
        env.render()

        if done:
            if env.winner == -1:
                print("It's a draw!")
            elif env.winner == 0:
                print("MCTS Agent wins!")
            else:
                print("You win!")
            break


if __name__ == "__main__":
    # Step 1: Train the MCTS agent
    #train_mcts(num_episodes=3200, num_simulations=1000)

    # Load the trained Q-table
    with open('q_table.pkl', 'rb') as f:
        q_table1 = pickle.load(f)
    
    with open('qtables/q_table.pkl', 'rb') as f:
        q_table2 = pickle.load(f)
    #with open('qtable.json', 'w') as f:
    #    import json
    #    json.dump({str(k): v for k, v in q_table.items()}, f, indent=4)


    # Step 2: Play against the trained MCTS agent
    # human_vs_mcts(q_table)\
    mcts_vs_mcts(q_table1, q_table2)
