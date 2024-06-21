from connect4_bitboards import Connect4BitboardEnv


env = Connect4BitboardEnv()

results = [(4398314950528, 35972464643)]

for result in results:
    env.board = list(result)
    env.render()
