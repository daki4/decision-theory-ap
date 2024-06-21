import copy
import pprint
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Connect4BitboardEnv(gym.Env):
    def __init__(self):
        super(Connect4BitboardEnv, self).__init__()
        self.board_height = 6
        self.board_width = 7
        self.size = self.board_height * self.board_width
        
        self.observation_space = spaces.Tuple((
            spaces.Discrete(1 << self.size),
            spaces.Discrete(1 << self.size)
        ))
        self.action_space = spaces.Discrete(self.board_width)
        
        self.reset()

    def clone(self):
        new = copy.deepcopy(self)
        # new = Connect4BitboardEnv()
        # new.board = self.board.copy()
        # new.board_height = self.board_height
        # new.board_width = self.board_width
        # new.size = self.size
        # new.current_player = self.current_player
        # new.heights = self.heights
        return new
        
    def reset(self):
        self.board = [0, 0]
        self.current_player = 0
        self.heights = [0] * self.board_width
        return self._get_obs()

    def step(self, action):
        if self.heights[action] >= self.board_height:
            raise ValueError("Invalid action: column is full")
        
        move = 1 << (action * (self.board_height + 1) + self.heights[action])
        self.board[self.current_player] ^= move
        self.heights[action] += 1

        done = self._check_win(self.board[self.current_player])
        reward = 1 if done else 0
        
        # Tie
        if not done and all(h == self.board_height for h in self.heights):
            done = True
            reward = 0.5
        
        self.current_player = 1 - self.current_player
        # self.render()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.board[0], self.board[1])

    def _check_win(self, board):
        directions = [1, self.board_height + 1, self.board_height, self.board_height + 2]
        for direction in directions:
            bb = board & (board >> direction)
            if bb & (bb >> (2 * direction)):
                return True
        return False

    def render(self, mode='human'):
        board_str = ''
        for row in range(self.board_height):
            for col in range(self.board_width):
                pos = col * (self.board_height + 1) + row
                if (self.board[0] >> pos) & 1:
                    board_str += 'X '
                elif (self.board[1] >> pos) & 1:
                    board_str += 'O '
                else:
                    board_str += '. '
            board_str += '\n'
        print(board_str)
    
    def get_possible_moves(self):
        return [col for col in range(self.board_width) if self.heights[col] < self.board_height]


    def generate_transition_probabilities(self, state, opponent_policy):
        board0, board1 = state
        current_player = self.current_player
        transition_probabilities = {}

        for action in range(self.board_width):
            if self.heights[action] >= self.board_height:
                continue
            
            new_board0, new_board1 = board0, board1
            move = 1 << (action * (self.board_height + 1) + self.heights[action])
            
            if current_player == 0:
                new_board0 ^= move
            else:
                new_board1 ^= move

            new_state = (new_board0, new_board1)
            done = self._check_win(new_board0 if current_player == 0 else new_board1)
            
            if done:
                transition_probabilities[action] = (new_state, 1.0)
            else:
                opponent_transition_probs = opponent_policy(self, new_state, self.heights, 1 - current_player)
                transition_probabilities[action] = (new_state, opponent_transition_probs)
        
        return transition_probabilities


# Usage example:
# env = Connect4BitboardEnv()
# obs = env.reset()
# env.render()

# # Simulate a game step
# print(env.board)
# obs, reward, done, info = env.step(3)
# # print(env.board)
# obs, reward, done, info = env.step(1)
# # print(env.board)
# # obs, reward, done, info = env.step(3)
# # print(env.board)
# obs, reward, done, info = env.step(3)
# # print(env.board)
# # print(env.step(3))
# # print(env.board)
# obs, reward, done, info = env.step(2)
# # print(env.board)
# # print(env.step(3))
# # print(env.board)
# obs, reward, done, info = env.step(3)
# obs, reward, done, info = env.step(3)
# obs, reward, done, info = env.step(3)
# obs, reward, done, info = env.step(3)
# print(env.get_possible_moves())
# # print(env.board)
# env.render()
# transition_probs = env.generate_transition_probabilities(obs, random_opponent_policy)
# transition_probs = env.generate_transition_probabilities(obs, lambda state, heights, player: minimax_opponent_policy(state, heights, player, depth=10))
# pprint.pprint(transition_probs)
