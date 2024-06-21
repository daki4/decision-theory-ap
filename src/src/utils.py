import numpy as np
from colorama import Fore

from environment_connect4 import Connect4Env
height = 7
width = 6
connect = 4

# def decode_board(board_string, rows=6, columns=7):
#     def reverse_lambda(value):
#         if value == 0:
#             return -1
#         elif value == -1:
#             return 1
#         elif value == 1:
#             return 0
#         else:
#             return value

#     board_list = list(map(int, board_string.split('|')))
#     if len(b) != rows * columns:
#         raise ValueError("Board string does not match the expected size for a Connect4 board.")
#     board = np.zeros((rows, columns), dtype=int)

#     for col in range(rows):
#         for row in range(columns):
#             board[col, row] = board_list[row * rows + col]
    
#     # board = list(np.rot90(board, k=-1))
#     print(board)
#     count_1 = np.count_nonzero(board == 1)
#     count_minus1 = np.count_nonzero(board == -1)
    
#     if count_1 > count_minus1:
#         turn = -1  # It's the turn of the player with -1 pieces
#     else:
#         turn = 1  # It's the turn of the player with 1 pieces

#     return board, turn

def decode_board(board_string, rows=6, columns=7):
    board_list = list(map(int, board_string.split('|')))
    if len(board_list) != rows * columns:
        raise ValueError("Board string does not match the expected size for a Connect4 board.")
    
    # Initialize the board with zeros
    board = np.zeros((columns, rows), dtype=int)
    reverse_lambda = lambda x: -1 if x == 0 else 0 if x == -1 else 1 if x == 1 else x
    board_list = np.array([reverse_lambda(v) for v in board_list])

    # Fill the board with values from the board list
    for idx, value in enumerate(board_list):
        row = idx // columns
        col = idx % columns
        board[col][row] = value

    board = np.rot90(board, k=-2)
    # Determine whose turn it is by counting the pieces
    count_1 = np.count_nonzero(board == 1)
    count_minus1 = np.count_nonzero(board == -1)
    
    if count_1 > count_minus1:
        turn = -1  # It's the turn of the player with -1 pieces
    else:
        turn = 1  # It's the turn of the player with 1 pieces
    print(board)
    return board, turn

def render(board, mode='human'):
    if mode != 'human': raise NotImplementedError('Rendering has not been coded yet')
    s = ""
    for x in range(height - 1, -1, -1):
        for y in range(width):
            s += {-1: Fore.WHITE + '.', 0: Fore.RED + 'X', 1: Fore.YELLOW + 'O'}[board[x][y]]
            s += Fore.RESET
        s += "\n"
    print(s)


def is_on_board(board, x, y):
    rows, columns = board.shape
    return 0 <= x < rows and 0 <= y < columns

def does_move_win(board, column, piece):
    # Find the first empty row in the given column
    rows, columns = board.shape
    row = next((r for r in range(rows-1, -1, -1) if board[r][column] == -1), None)
    
    if row is None:
        raise ValueError("Column is full")

    # Simulate dropping the piece
    board_copy = board.copy()
    board_copy[row][column] = piece

    # Check if this move wins the game
    for dx, dy in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
        p = 1  # positive direction
        while is_on_board(board_copy, row + p * dx, column + p * dy) and board_copy[row + p * dx][column + p * dy] == piece:
            p += 1
        n = 1  # negative direction
        while is_on_board(board_copy, row - n * dx, column - n * dy) and board_copy[row - n * dx][column - n * dy] == piece:
            n += 1

        if p + n >= 5:  # need 4 in a row, and p + n gives us the total count of connected pieces
            return True

    return False

def evaluate(state, action):
    decoded, cp = decode_board(state)
    # print('is game won: ', does_move_win(decoded, action, cp))
    env = Connect4Env()
    env.board = decoded
    # env.unstep(action)
    env.render()
    env.step(action)
    env.render()
# s = '0|0|0|0|0|0|0|0|0|0|0|0|-1|0|0|0|0|0|0|1|0|0|0|0|0|0|-1|-1|1|-1|0|-1|0|1|-1|1|1|1|-1|1|-1|1'
# a = 4
s = '0|0|0|0|0|0|0|0|0|0|0|0|0|1|1|0|0|0|0|0|-1|-1|0|0|0|0|0|-1|1|-1|0|-1|0|1|-1|1|1|1|-1|0|-1|1'
a = 0
evaluate(s, a)
