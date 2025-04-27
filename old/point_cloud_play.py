from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import visualiser

MAX_POINTS = 2000
POINTS_PER_FRAME = 500
MAX_RAY_LENGTH = 5000

clear_sm64_exes()

game = SM64_GAME(server=True, server_port=7777)
points_array = np.array([[0,0,0]])
normals_array = np.array([[0,0,0]])

with tqdm.tqdm() as pbar:
    while True:
        stickX = random.randint(-80, 80)
        stickY = random.randint(-80, 80)
        buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
        # buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]

        game.step_game(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        # marioState = game.get_mario_state(0)
        # networkPlayer = game.get_network_player(0)
        marioStates = [game.get_mario_state(i) for i in range(2)]

        start_time = time.time()

        new_pts, new_normals = game.get_raycast_sphere_with_normal(amount=POINTS_PER_FRAME, maxRayLength=MAX_RAY_LENGTH)

        # Filter out invalid points
        zero_normals = np.all(new_normals == 0, axis=1)
        new_normals = new_normals[~zero_normals]
        new_pts = new_pts[~zero_normals]

        points_array = np.concatenate((points_array, new_pts), axis=0)
        points_array = points_array[-MAX_POINTS:]

        normals_array = np.concatenate((normals_array, new_normals), axis=0)
        normals_array = normals_array[-MAX_POINTS:]

        visualiser.visualise_game(marioStates, points_array, normals_array)

        pbar.update(1)
