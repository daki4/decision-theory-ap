
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

class TDLearningAgent:
    def __init__(self, env, alpha=0.2, gamma=0.9, epsilon=0.5, min_epsilon=0.1, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay # decay rate
        self.min_epsilon = min_epsilon
        self.value_table = {}  # State value table

    def get_state_value(self, state):
        state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
        if state_tuple not in self.value_table:
            self.value_table[state_tuple] = 0
        return self.value_table[state_tuple]

    def update_state_value(self, state, player, reward, next_state):
        state_tuple = tuple(tuple(st.flatten().tolist()) for st in state)
        next_state_value = self.get_state_value(next_state)
        current_state_value = self.get_state_value(state)
        self.value_table[state_tuple] += self.alpha * (reward[player] + self.gamma * next_state_value - current_state_value)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_moves())
        else:
            values = []
            for action in self.env.get_moves():
                next_state, _, _, _ = self.env.clone().step(action)
                values.append(self.get_state_value(next_state))
            return self.env.get_moves()[np.argmax(values)]

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action()
                next_state, reward, done, _ = self.env.step(action)
                self.update_state_value(state, self.env.current_player, reward, next_state)
                state = next_state
                # play for the other player
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                if not done:
                    _, _, done, _ = self.env.step(np.random.choice(self.env.get_moves()))
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes} completed.")

def train_agent(episodes):
    env = Connect4Env()
    agent = TDLearningAgent(env)
    agent.train(episodes)
    return agent.value_table

def merge_value_tables(value_tables):
    merged_table = {}
    for table in value_tables:
        for key, value in table.items():
            if key in merged_table:
                merged_table[key] += value
            else:
                merged_table[key] = value
    return merged_table

if __name__ == '__main__':
    num_processes = mp.cpu_count()
    episodes_per_process = (320_000  * 10)// num_processes

    with mp.Pool(num_processes) as pool:
        value_tables = pool.map(train_agent, [episodes_per_process] * num_processes)

    merged_value_table = merge_value_tables(value_tables)
    
    with open('qtable.out', 'w') as f:
        import json
        json.dump({str(k): v for k, v in merged_value_table.items()}, f, indent=4)

