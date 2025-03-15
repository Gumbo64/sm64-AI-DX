
from sm64env.sm64env_nothing import SM64_ENV_NOTHING


import numpy as np
import torch
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def execute_path(path, frame_time=0):
    _, info = env.reset()
    print("STARTING")
    position = info['pos']
    for action in path:
        stick_actions = action[0:2] * 80
        button_actions = action[2:5] > 0.95
        action = (stick_actions, button_actions)

        obs, reward, done, truncated, info = env.step(action)
        position = info['pos']
        time.sleep(frame_time)

length = 800
multi_step = 4

env = SM64_ENV_NOTHING(multi_step=multi_step, server=True, server_port=7777)


# path = np.load('best_path_30014553.703223668.npy')
# path = np.load('best_path/100_2488.2315575332086.npy')
path = np.load('best_path_39062318.86274888.npy')
while True:
    execute_path(path, frame_time=0.0166 * multi_step)