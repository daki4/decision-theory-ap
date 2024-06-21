
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
from itertools import chain

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
from multiprocessing import Pool, cpu_count, Manager

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

        return root.most_visited_child().move

    def update_q_table(self, node):
        while node:
            state_str = ''.join(map(str, node.state.board.flatten()))
            key = (state_str, node.move)
            if key not in self.q_table:
                self.q_table[key] = 0
            self.q_table[key] += node.value
            node = node.parent

episode = 0
def get_state_action_key(state, action):
    state_str = ''.join(map(str, state.flatten()))
    return (state_str, action)


def run_episode(args):
    global episode
    episode += 1
    q_table, num_simulations = args
    env = Connect4Env()
    mcts = MCTS(env, q_table, num_simulations=num_simulations)

    state = env.reset()
    done = False
    while not done:
        if env.current_player == 0:
            action = mcts.search(env.clone())
        else:
            possible_moves = env.get_moves()
            action = random.choice(possible_moves)

        next_state, reward, done, info = env.step(action)

        key = get_state_action_key(state, action)
        if key not in q_table:
            q_table[key] = 0

        q_table[key] += reward[env.current_player]
        state = next_state

        if done:
            break
    print(f'episode {episode} done')

    return q_table

def train_mcts(num_episodes=100, num_simulations=1000):
    with Manager() as manager:
        q_table = manager.dict()

        with Pool(processes=cpu_count()) as pool:
            q_tables = pool.map(run_episode, [(q_table, num_simulations) for _ in range(num_episodes)])

        q_table = dict(q_table)  # Convert manager.dict() to a regular dictionary

        with open('q_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    print("Q-table generated and saved to 'q_table.pkl'")

## minimax
import math
import multiprocessing as mp

def minimax(node, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or node.winner is not None:
        return node.get_result(maximizingPlayer)
    
    valid_moves = node.get_moves()
    if maximizingPlayer:
        value = -math.inf
        for move in valid_moves:
            child = node.clone()
            child.step(move)
            value = max(value, minimax(child, depth-1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for move in valid_moves:
            child = node.clone()
            child.step(move)
            value = min(value, minimax(child, depth-1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def minimax_worker(child, depth, alpha, beta, maximizingPlayer, return_dict, idx):
    return_dict[idx] = minimax(child, depth, alpha, beta, maximizingPlayer)

def minimax_decision(env, depth=4):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    best_value = -math.inf
    best_move = None

    valid_moves = env.get_moves()
    for idx, move in enumerate(valid_moves):
        child = env.clone()
        child.step(move)
        p = mp.Process(target=minimax_worker, args=(child, depth-1, -math.inf, math.inf, False, return_dict, idx))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    for idx, move in enumerate(valid_moves):
        value = return_dict[idx]
        if value > best_value:
            best_value = value
            best_move = move

    return best_move


def minimax_vs_mcts(q_table, minimax_depth=4):
    env = Connect4Env()
    mcts = MCTS(env, q_table, num_simulations=1000)

    state = env.reset()
    done = False

    print("Starting a game of Connect 4!")
    env.render()

    while not done:
        if env.current_player == 0:
            print("MCTS Agent's turn:")
            action = mcts.search(env.clone())
        else:
            print("Minimax Agent's turn:")
            action = minimax_decision(env.clone(), minimax_depth)

        state, reward, done, info = env.step(action)
        env.render()

        if done:
            if env.winner == -1:
                print("It's a draw!")
            elif env.winner == 0:
                print("MCTS Agent wins!")
            else:
                print("Minimax Agent wins!")
            break



### Step 2: Playing Phase

def human_vs_mcts(q_table):
    env = Connect4Env()
    mcts = MCTS(env, q_table, num_simulations=1000)

    state = env.reset()
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

        state, reward, done, info = env.step(action)
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
    # train_mcts(num_episodes=100, num_simulations=1000)

    # Load the trained Q-table
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)

    # with open('qtable.json', 'w') as f:
    #     import json
    #     json.dump({str(k): v for k, v in q_table.items()}, f, indent=4)


    # Step 2: Play against the trained MCTS agent
    # human_vs_mcts(q_table)\
    minimax_vs_mcts(q_table, minimax_depth=4)
