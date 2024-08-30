from load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time

clear_sm64_exes()

game = SM64_GAME(server=True, server_port=7777)


with tqdm.tqdm() as pbar:
    while True:
        stickX = random.randint(-80, 80)
        stickY = random.randint(-80, 80)
        buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
        # buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]
        game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        game.step_game()
        pbar.update(1)