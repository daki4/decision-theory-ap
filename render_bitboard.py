from connect4_bitboards import Connect4BitboardEnv


env = Connect4BitboardEnv()

results = [(331943, 175432),
(430243, 77132),
(331943, 175432),
(430243, 77132),
(430243, 77132)]

for result in results:
    env.board = list(result)
    env.render()
