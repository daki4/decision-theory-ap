import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
from colorama import Fore
import random

class Connect4Env(gym.Env):

    def __init__(self, width=7, height=6, connect=4):
        self.num_players = 2

        self.width = width
        self.height = height
        self.connect = connect

        player_observation_space = Box(low=-1, high=1,
                                       shape=(self.width, self.height),
                                       dtype=np.int32)
        self.observation_space = player_observation_space
        self.action_space = Tuple([Discrete(self.width) for _ in range(self.num_players)])

        self.state_space_size = (self.height * self.width) ** 3

        self.reset()

    def reset(self):
        """
        Initialises the Connect 4 gameboard.
        """
        self.board = np.full((self.width, self.height), -1)

        self.current_player = 0
        self.winner = None
        return self.get_player_observations()

    #  0 - empty
    # -1 - one player
    #  1 - another player
    def get_player_observations(self):
        transformed_array = np.array([list(map(lambda x: 0 if x == -1 else -1 if x == 0 else 1 if x == 1 else x, row)) for row in self.board])
    
        transposed_array =  np.rot90(transformed_array, k=1)
    
        return transposed_array

    def step(self, movecol):
        """
        Applies a move by a player to the game board in a format which is suitable for adversary play
        """
        if not(movecol >= 0 and movecol <= self.width and self.board[movecol][self.height - 1] == -1):
            raise IndexError(f'Invalid move. tried to place a chip on column {movecol} which is already full. Valid moves are: {self.get_moves()}')
        row = self.height - 1
        while row >= 0 and self.board[movecol][row] == -1:
            row -= 1

        row += 1

        self.board[movecol][row] = self.current_player
        self.current_player = 1 - self.current_player

        self.winner, reward_vector = self.check_for_episode_termination(movecol, row)

        info = {'legal_actions': self.get_moves(),
                'current_player': self.current_player}
        return self.get_player_observations(), reward_vector, \
               self.winner is not None, info

    def check_for_episode_termination(self, movecol, row):
        """
        Check for victories in the current state and generate rewards for the state
        """
        winner, reward_vector = self.winner, [0, 0]
        if self.does_move_win(movecol, row):
            winner = 1 - self.current_player
            if winner == 0: reward_vector = [1, -1]
            elif winner == 1: reward_vector = [-1, 1]
        elif self.get_moves() == []:  # A draw has happened
            winner = -1
        return winner, reward_vector

    def clone(self):
        st = Connect4Env(width=self.width, height=self.height)
        st.current_player = self.current_player
        st.winner = self.winner
        st.board = np.array([self.board[col][:] for col in range(self.width)])
        return st

    def get_moves(self):
        """
        :returns: array with columns where there is a possible move
        """
        if self.winner is not None:
            return []
        return [col for col in range(self.width) if self.board[col][self.height - 1] == -1]

    def does_move_win(self, x, y, me=None):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        """
        if me is None:
            me = self.board[x][y]
        for dx, dy in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1 # positive direction
            while self.is_on_board(x+p*dx, y+p*dy) and self.board[x+p*dx][y+p*dy] == me:
                p += 1
            n = 1 # negative direction
            while self.is_on_board(x-n*dx, y-n*dy) and self.board[x-n*dx][y-n*dy] == me:
                n += 1

            if p + n >= (self.connect + 1): # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
                return True

        return False
    
    # swaps one random token of player A with another random token of player B 
    def swap_random_tokens(self):
        while True:
            old_col = random.choice(range(self.height))
            old_row = random.choice(range(self.width))
            old_tile_val = self.board[old_row][old_col]

            # print("old_col: ", old_col)
            # print("old_row: ", old_row)
            # print("old_tile_val: ", old_tile_val)

            if old_tile_val == -1:

                # print("old_tile_val is -1")

                continue
        
            new_col = random.choice(range(self.height))
            new_row = random.choice(range(self.width))
            new_tile_val = self.board[new_row][new_col]

            # print("new_col: ", new_col)
            # print("new_row: ", new_row)
            # print("new_tile_val: ", new_tile_val)


            if new_tile_val == -1 or new_tile_val == old_tile_val:

                # print("new tile val is -1 or same as the other")

                continue

            if self.does_move_win(new_row, new_col, me=old_tile_val):

                # print("move wins")

                continue

            if self.does_move_win(old_row, old_col, me=new_tile_val):
                
                # print("move wins")

                continue
            
            # print("yeeeeeee")

            self.board[old_row][old_col], self.board[new_row][new_col] = self.board[new_row][new_col], self.board[old_row][old_col]
            break

    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        if self.winner == -1: return 0  # A draw occurred
        return +1 if player == self.winner else -1

    def render(self, mode='human'):
        if mode != 'human': raise NotImplementedError('Rendering has not been coded yet')
        s = ""
        for x in range(self.height - 1, -1, -1):
            for y in range(self.width):
                s += {-1: Fore.WHITE + '.', 0: Fore.RED + 'X', 1: Fore.YELLOW + 'O'}[self.board[y][x]]
                s += Fore.RESET
            s += "\n"
        print(s)

    def unstep(self, column):
        """
        Removes the latest player token from a specific column on the board.
        """
        width, height = self.board.shape
        
        for row in range(height - 1, -1, -1):
            if self.board[column][row] != -1:  # Assuming -1 represents an empty slot
                self.board[column][row] = -1
                return
        return

