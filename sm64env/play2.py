from load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time

clear_sm64_exes()

games = []
games.append(SM64_GAME(server=True, server_port=7777))

with tqdm.tqdm() as pbar:
    i = 1
    while True:
        if i % 1 == 0 and len(games) < 16:
            games.append(SM64_GAME(server=False, server_port=7777))
            i = 0
        i += 1

        for game in games:
            # time.sleep(0.2)
            stickX = random.randint(-80, 80)
            stickY = random.randint(-80, 80)
            # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
            # game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
            game.set_controller(stickX=stickX, stickY=stickY)
            game.step_game()
            
        pbar.update(1)