from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time
import numpy as np
import math

clear_sm64_exes()
TREE_FACTOR = 2
NUM_GAMES = 15
ALL_SERVER = True

games = [SM64_GAME(server = (i == 0), server_port=7777) for i in range(NUM_GAMES)]
# games = [SM64_GAME(server = False, server_port=7777) for i in range(NUM_GAMES)]

with tqdm.tqdm() as pbar:
    while True:
        for i, game in enumerate(games):
            if i == 0:
                # pretend that index 0 is at the bottom of the tree
                if (len(games) == 1):
                    goalIndex = 1
                else:
                    goalIndex = len(games) // TREE_FACTOR
            elif i == 1:
                # Top of the tree, points to the player
                goalIndex = len(games)

            else:
                goalIndex = i // TREE_FACTOR

            # Using the server's global index system
            # goalIndex = len(games)

            pos = games[i].get_mario_state(0).pos
            if goalIndex == len(games):
                # goalPos = games[0].get_mario_state(len(games)).pos
                # print(np.array(goalPos))
                goalPos = np.array([5524.7285, 3072, 1187.4833])
            else:
                goalPos = games[goalIndex].get_mario_state(0).pos
            
            diff = np.array(pos) - np.array(goalPos)

            angle = math.atan2(diff[2], diff[0])

            # stickX = -math.cos(angle) * 80 * min(1, abs(diff[0])/200)
            # stickY = math.sin(angle) * 80 * min(1, abs(diff[2])/200)
            stickX = -math.cos(angle) * 80
            stickY = math.sin(angle) * 80
            buttonA = (random.random() <= 1/5) if diff[1] < -220 else (random.random() <= 1/50)
            # buttonA = (random.random() <= 1/5) if diff[1] < -220 else 0

            game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA)
            game.step_game()         
        # pbar.update(1)