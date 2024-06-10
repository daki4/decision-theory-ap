import gymnasium as gym

env = gym.make('Connect4-v0')
env.reset()
env.unwrapped.board = \
   [[-1, -1, -1, -1, -1, -1],
    [ 1,  1, -1, -1, -1, -1],
    [ 1,  1,  0,  0, -1, -1],
    [ 1,  0,  0, -1, -1, -1],
    [ 1,  1,  1, -1, -1, -1],
    [ 1,  0,  1, -1, -1, -1],
    [ 0,  0, -1, -1, -1, -1]]
