from load_sm64_CDLL import SM64_GAME


game = SM64_GAME(server=False, server_port=7777)


while True:
    game.step_game()