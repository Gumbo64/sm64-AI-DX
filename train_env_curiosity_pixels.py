import gym.vector
import sm64env.curiosity_util as curiosity_util
from sm64env.sm64env_pixels import SM64_ENV_PIXELS
from sm64env.load_sm64_CDLL import clear_sm64_exes
from visualiser import visualise_game_tokens, visualise_curiosity
import gym
import time

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
import numpy as np

import wandb
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
clear_sm64_exes()

n_envs = 8
steps_per_iter = 1600
ppo_epochs = 4

mini_batch_size = 256

iter_per_log = 1
iter_per_save = 10


frame_stack_amount = 4
height = 72
width = 128

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

obs_stack = np.zeros((n_envs, frame_stack_amount, height, width), dtype=np.float32)
def format_obs(obs):
    global obs_stack
    obs = np.dot(obs, [0.30, 0.59, 0.11]) / 255
    obs_stack = np.roll(obs_stack, shift=1, axis=1)
    obs_stack[:, 0] = obs
    
    return obs_stack

def curiosity_reward(positions, curiosity):
    curiosity.add_circles(positions)
    visits = curiosity.get_visits_multi(positions)
    # rewards = np.exp(- 4 * visits / curiosity.max_visits)
    rewards = 1 - (visits / curiosity.max_visits)
    return rewards

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.mid_size = 2048
        self.num_outputs = 5

        self.std = 0.0

        self.conv = nn.Sequential(
            # 4 frame stack so that is the first number
            layer_init(nn.Conv2d(frame_stack_amount, 64, 8, stride=2)),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 256, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(256, 512, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(512, 512, 2, stride=1)),
            nn.Flatten(),


            # input size calculated from torch_layer_size_test.py, given frame_stack_amount channels and 128x72 input
            layer_init(nn.Linear(2048, 2048)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(2048, 2048)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(2048, self.mid_size)),
            nn.LeakyReLU(),
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.mid_size, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.mid_size, 2048)),
            nn.Tanh(),
            layer_init(nn.Linear(2048, 1024)),
            # layer_init(nn.Linear(512, self.num_outputs), std=actor_std),
            layer_init(nn.Linear(1024, self.num_outputs)),
            nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.ones(1, self.num_outputs) * self.std)

    def forward(self, x):
        x = self.conv(x)
        # Actor-critic part
        value = self.critic(x)
        mu = self.actor_mean(x)
        std   = self.actor_log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    

def make_env(i):
    def mkenv():
        return SM64_ENV_PIXELS(server = True, server_port=7777 + i, multi_step=4)
    return mkenv

# https://github.com/higgsfield-ai/higgsfield/blob/main/higgsfield/rl/rl_adventure_2/3.ppo.ipynb
def ppo_iter(mini_batch_size, obs_s, actions, log_probs, returns, advantage):
    batch_size = len(obs_s)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        obs_s_yield = torch.zeros((mini_batch_size, frame_stack_amount, height, width), dtype=torch.float32).to(device)
        for i in range(frame_stack_amount):
            valid_ids = rand_ids - i >= 0
            obs_s_yield[valid_ids, i] = obs_s[rand_ids[valid_ids] - i]
         
        yield obs_s_yield, actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(ppo_epochs, mini_batch_size, obs_s, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for obs, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, obs_s, actions, log_probs, returns, advantages):
            # already torchified
            dist, value = agent.forward(obs)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            # print(torch.cuda.get_device_properties(0).total_memory,torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0))
            optimizer.step()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def clamp_stick(tensor_vec):
    # Clamp the stick length, but maintain the direction
    vec_length = torch.norm(tensor_vec, dim=1, keepdim=True)
    norm_vec = torch.div(tensor_vec, vec_length)
    clamped_length = torch.abs(torch.clamp(vec_length, -80, 80))
   
    return norm_vec * clamped_length



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent().to(device)

# agent.load_state_dict(torch.load("ppo_1728480135.3339143_40.pth"))
# agent.load_state_dict(torch.load("ppo_1728480135.3339143_220.pth"))
# agent.load_state_dict(torch.load("ppo_1728491932.1914167_3090.pth"))
# agent.load_state_dict(torch.load("ppo_1728754329.291212_340.pth"))
# agent.load_state_dict(torch.load("ppo_1728804252.0307822_160.pth"))
# agent.load_state_dict(torch.load("ppo_1728814183.1124492_110.pth"))
# agent.load_state_dict(torch.load("ppo_1728877177.6916175_150.pth"))
# agent.load_state_dict(torch.load("models/small_ppo_1734886878.8964658_1100.pth"))
# agent.actor_log_std.data.fill_(0)

# optimizer = optim.Adam(agent.parameters(), lr=3e-4, weight_decay=1e-4)
optimizer = optim.Adam(agent.parameters(), lr=3e-4)

run_name = f"small_ppo_{time.time()}"
wandb.init(
    project="sm64env",
    sync_tensorboard=True,
    name=run_name,
    monitor_gym=True,
    save_code=True,
)
writer = SummaryWriter(f"runs/{run_name}")

print(torch.cuda.get_device_properties(0).total_memory,torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0))
# print("Parameters: ", sum(p.numel() for p in agent.parameters()))
# print("Size in GB: ", sum(p.numel() for p in agent.parameters()) * 4 / 1024 / 1024 / 1024)
# torch.save(agent.state_dict(), "agent.pth")

envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
# envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])
curiosity = curiosity_util.CURIOSITY(max_visits=400)


last_log_time = time.time()
iter = 1
with tqdm.tqdm() as iterbar:
    while True:
        log_probs = []
        values    = []
        obs_s    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        obs, info = envs.reset()

        curiosity.soft_reset()

        obs_stack = np.zeros((n_envs, frame_stack_amount, height, width), dtype=np.float32)
        obs = format_obs(obs)

        for _ in tqdm.tqdm(range(steps_per_iter), leave=False):
            # print(torch.cuda.get_device_properties(0).total_memory,torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0))

            # Feed forward
            with torch.no_grad():
                torchObs = torch.tensor(obs, dtype=torch.float32).to(device)
                dist, value = agent.forward(torchObs)

            # Step
            action_raw = dist.sample()
            stick_actions = clamp_stick(action_raw[:, 0:2] * 80).cpu()

            button_actions = (action_raw[:, 2:5] > 0.8).bool().cpu()
            action = (stick_actions, button_actions)
            next_obs, _, done, truncated, info =  envs.step(action)
            next_obs = format_obs(next_obs) # grayscale + stack frames

            positions = np.array([info['pos'][i] for i in range(n_envs)])
            reward = curiosity_reward(positions, curiosity) # curiosity reward

            # Calculate storage stuff
            log_prob = dist.log_prob(action_raw)
            entropy += dist.entropy().mean()

            # Storage
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device))
            masks.append(torch.tensor(1 - done, dtype=torch.float32).unsqueeze(1).to(device))
            
            obs_s.append(torchObs[:,0]) #all the internals of the torchObs are already on the device
            actions.append(action_raw)

            obs = next_obs
            #logging here

        # visualise_curiosity(envs.get_attr('curiosity')[0])

        # Bootstrapping the last obs
        with torch.no_grad():
            last_torchObs = torch.tensor(obs, dtype=torch.float32).to(device)
            _, last_value = agent.forward(last_torchObs)
            returns = compute_gae(last_value, rewards, masks, values)

        # Cat and detach. Each players' experiences are now on the same dimension
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        obs_s = torch.cat(obs_s)
        actions   = torch.cat(actions).detach()
        advantages = returns - values

        ppo_update(ppo_epochs, mini_batch_size, obs_s, actions, log_probs, returns, advantages)

        iter += 1
        iterbar.update(1)
        if iter % iter_per_save == 0:
            os.makedirs(f"models/{run_name}", exist_ok=True)
            torch.save(agent.state_dict(), f"models/{run_name}/{iter}.pth")

        if iter % iter_per_log == 0:
            rewards = torch.cat(rewards).detach()

            step_count = iter * steps_per_iter * n_envs
            writer.add_scalar("Entropy", entropy, step_count)
            writer.add_scalar("Advantage", advantages.mean(), step_count)
            writer.add_scalar("Value", values.mean(), step_count)
            writer.add_scalar("Return", returns.mean(), step_count)
            writer.add_scalar("Average reward", rewards.mean(), step_count)
            writer.add_scalar("SPS", iter_per_log * steps_per_iter * n_envs / (time.time() - last_log_time), step_count)
            last_log_time = time.time()
        # envs.close()
        # clear_sm64_exes()











