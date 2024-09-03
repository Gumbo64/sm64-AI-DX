from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time

clear_sm64_exes()

NUM_GAMES = 2
ALL_SERVER = False

games = [SM64_GAME(server = (i == 0 or ALL_SERVER), server_port=7777) for i in range(NUM_GAMES)]

with tqdm.tqdm() as pbar:
    while True:
        for i, game in enumerate(games):
            # stickX = random.randint(-80, 80)
            # stickY = random.randint(-80, 80)
            stickX = 0
            stickY = 0
            buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
            # game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
            # game.set_controller(stickX=stickX, stickY=stickY)
            game.step_game()
            # state = game.get_mario_state(0)
            # pos = state.pos
            # print([int(pos[i]) for i in range(3)])
        pbar.update(1)