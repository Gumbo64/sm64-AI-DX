from sm64env.load_sm64_CDLL import clear_sm64_exes
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
import tqdm
import random
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import visualiser

# POINTS_PER_FRAME = 50000
# FPS_AMOUNT = 20
# POINTS_PER_FRAME = 50
# FPS_AMOUNT = None
# num_points=1000, fps_amount = 300, max_ray_length=10000,
POINTS_PER_FRAME = 1000
FPS_AMOUNT = 10000
MAX_RAY_LENGTH = 8000

clear_sm64_exes()

game = SM64_ENV_CURIOSITY(server=True, server_port=7777, fps_amount=FPS_AMOUNT, num_points=POINTS_PER_FRAME, max_ray_length=MAX_RAY_LENGTH)

# with tqdm.tqdm() as pbar:
while True:
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
    # buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]
    # stickX = 0
    stickY = 80
    buttonA, buttonB, buttonZ = 0, 0, 0

    action = [(stickX, stickY), (buttonA, buttonB, buttonZ)]
    obs, reward, done, truncated, info = game.step(action)
    visualiser.visualise_game_tokens(obs)
    print(obs[0][3:9])
    # pbar.update(1)
