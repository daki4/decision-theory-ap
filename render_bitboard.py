from connect4_bitboards import Connect4BitboardEnv


env = Connect4BitboardEnv()

results = [(129, 2097152)]

for result in results:
    env.board = list(result)
    env.render()
