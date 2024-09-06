from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time
import numpy as np
import math
import visualiser

clear_sm64_exes()
TREE_FACTOR = 2
NUM_GAMES = 16
ALL_SERVER = True

MAX_POINTS = 10000
POINTS_PER_FRAME = MAX_POINTS // NUM_GAMES
MAX_RAY_LENGTH = 5000


points_array = np.array([[0,0,0]])
normals_array = np.array([[0,0,0]])

visualiser.visualise_game([], points_array, normals_array)
games = []
for i in range(NUM_GAMES):
    games.append(SM64_GAME(server = (i == 0), server_port=7777))
    games[-1].step_game()
    print(f"Game {i} started")
    # time.sleep(0.5)
games = [games[0], games[-1]] + games[1:-1]

with tqdm.tqdm() as pbar:
    while True:
        marioStates = []
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
            newMarioState = games[i].get_mario_state(0)
            marioStates.append(newMarioState)

            pos = newMarioState.pos
            if goalIndex == len(games):
                # goalPos = games[0].get_mario_state(len(games)).pos
                goalPos = np.array([5524.7285, 3072, 1187.4833])
            else:
                # The goal can either be the player or a predefined position
                goalPos = games[goalIndex].get_mario_state(0).pos
                
            
            
            diff = np.array(pos) - np.array(goalPos)
            angle = math.atan2(diff[2], diff[0])
            stickX = -math.cos(angle) * 80
            stickY = math.sin(angle) * 80
            buttonA = (random.random() <= 1/5) if diff[1] < -220 else (random.random() <= 1/50)

            game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA)
            game.step_game()

            new_pts, new_normals = game.get_raycast_sphere_with_normal(amount=POINTS_PER_FRAME, maxRayLength=MAX_RAY_LENGTH)
            zero_normals = np.all(new_normals == 0, axis=1)
            new_normals = new_normals[~zero_normals]
            new_pts = new_pts[~zero_normals]

            points_array = np.concatenate((points_array, new_pts), axis=0)
            points_array = points_array[-MAX_POINTS:]

            normals_array = np.concatenate((normals_array, new_normals), axis=0)
            normals_array = normals_array[-MAX_POINTS:]

        # marioStates = [game.get_mario_state(len(games))] + marioStates
        marioStates = [marioStates[-1]] + marioStates
        visualiser.visualise_game(marioStates, points_array, normals_array)
        pbar.update(1)