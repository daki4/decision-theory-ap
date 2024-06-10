import gymnasium as gym

env = gym.make('Connect4-v0')
env.reset()

# -1 is empty
#  0 is Player A
#  1 is Player B
#  2 is pacman
env.unwrapped.board = \
   [[-1, -1, -1, -1, -1, -1],
    [ 1,  1, -1, -1, -1, -1],
    [ 1,  1,  0,  0, -1, -1],
    [ 1,  0,  0,  1,  1, -1],
    [ 1,  1,  1,  0,  0, -1],
    [ 1,  1,  2,  1,  1, -1],
    [ 0,  0,  1,  1,  1, -1]]


