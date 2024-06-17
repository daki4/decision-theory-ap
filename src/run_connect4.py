import environment_connect4
import numpy as np
env = environment_connect4.Connect4Env()

env.reset()

# -1 is empty
#  0 is Player A
#  1 is Player B
#  2 is pacman
env.board = \
   np.array(
    [[-1, -1, -1, -1, -1, -1],
    [ 1,  1, -1, -1, -1, -1],
    [ 1,  1,  0,  0, -1, -1],
    [ 1,  0,  0,  1,  1, -1],
    [ 1,  1,  1,  0,  0, -1],
    [ 1,  1,  2,  1,  1, -1],
    [ 0,  0,  1,  1,  1, -1]])


