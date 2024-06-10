from colorama import Fore
height = 7
width = 6
board = ((1, -1, -1, 1, 0, 0, 0), (-1, 1, 1, -1, 0, 0, 0), (1, 1, -1, -1, 0, 0, 0), (-1, -1, 1, 1, 0, 0, 0), (1, 1, -1, -1, -1, 0, 0), (-1, -1, 1, 1, 1, 0, -1))


def render(mode='human'):
        if mode != 'human': raise NotImplementedError('Rendering has not been coded yet')
        s = ""
        for x in range(height - 1, -1, -1):
            for y in range(width):
                s += {0: Fore.WHITE + '.', -1: Fore.RED + 'X', 1: Fore.YELLOW + 'O'}[board[y][x]]
                s += Fore.RESET
            s += "\n"
        print(s)


def is_game_won():
    rows = len(board)
    cols = len(board[0])
    
    def check_direction(start_row, start_col, delta_row, delta_col):
        piece = board[start_row][start_col]
        if piece == 0:
            return False
        
        for i in range(1, 4):
            row = start_row + delta_row * i
            col = start_col + delta_col * i
            if row < 0 or row >= rows or col < 0 or col >= cols or board[row][col] != piece:
                return False
        
        return True
    
    for row in range(rows):
        for col in range(cols):
            if col + 3 < cols and check_direction(row, col, 0, 1):
                return True
            if row + 3 < rows and check_direction(row, col, 1, 0):
                return True
            if row + 3 < rows and col + 3 < cols and check_direction(row, col, 1, 1):
                return True
            if row + 3 < rows and col - 3 >= 0 and check_direction(row, col, 1, -1):
                return True
    
    return False




print(board)
render()
print(is_game_won())