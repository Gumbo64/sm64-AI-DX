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
import numpy as np

n_envs = 3
clear_sm64_exes()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_inputs = 10
        self.num_outputs = 5
        actor_std = 0.01

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(64, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(64, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.num_outputs), std=actor_std),
        )
        self.actor_log_std = nn.Parameter(torch.ones(1, self.num_outputs) * actor_std)

    def forward(self, array_of_obs):
        xs = []
        for i in range(len(array_of_obs)):
            x = torch.tensor(array_of_obs[i], dtype=torch.float32)
            x = torch.nn.functional.pad(x, (0, 64 - self.num_inputs))
            x = self.transformer(x)
            x = x.mean(dim=0)
            xs.append(x)

        xs = torch.stack(xs)
        value = self.critic(xs)
        mu = self.actor_mean(xs)
        std   = self.actor_log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def make_env(i):
    def mkenv():
        # print(i, i % 16 == 0, 7777 + (i // 16))
        return SM64_ENV_CURIOSITY(multi_step=4, num_points=5000, 
                                  server= (i % 16 == 0), server_port=(7777 + (i // 16)))
        # return SM64_ENV_CURIOSITY(multi_step=multi_step, server=True, server_port=7777 + i)
    return mkenv



agent = Agent()
# print("Parameters: ", sum(p.numel() for p in agent.parameters()))
# print("Size in GB: ", sum(p.numel() for p in agent.parameters()) * 4 / 1024 / 1024 / 1024)
# torch.save(agent.state_dict(), "agent.pth")

# envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])
# env = SM64_ENV_CURIOSITY()

obs, info = envs.reset()
start_time = time.time()
while True:
    dist, value = agent.forward(obs)
    actions_raw = dist.sample()
    stick_actions = torch.clamp(actions_raw[:, 0:2], -1, 1) * 80
    button_actions = (actions_raw[:, 2:5] > 0).bool()
    actions = (stick_actions, button_actions)

    envs.step(actions)





