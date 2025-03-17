import gym.vector
import gym.vector
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from sm64env.load_sm64_CDLL import clear_sm64_exes
from visualiser import visualise_game_tokens, visualise_curiosity
import random
import gym
import time

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
import torch.cuda as cuda
import numpy as np

import wandb
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
import math
# os.environ["WANDB_SILENT"] = "true"

clear_sm64_exes()


import numpy as np
import torch
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt



length = 800
multi_step = 1

env = SM64_ENV_CURIOSITY(multi_step=multi_step, server=True, server_port=7777, fps_amount=1000, num_points=1000, max_visits=100000)

obs, info = env.reset()
while True:
    
    location_weights = obs[:, 7]
    location_weights = - np.exp(location_weights/env.max_visits)
    locations = obs[:, 0:3]
    normalised_locations = np.linalg.norm(locations, axis=0)

    weighted_dir = locations * location_weights[:, None]
    weighted_dir = weighted_dir.sum(axis=0)
    
    stickX = weighted_dir[0] * 80
    stickZ = weighted_dir[2] * 80

    print(stickX, stickZ)


    obs, reward, done, truncated, info = env.step(([stickX, stickZ], [0, 0, 0]))
    








