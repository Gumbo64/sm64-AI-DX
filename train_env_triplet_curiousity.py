import gym.vector
import gym.vector
from sm64env.sm64env_pixels_save import SM64_ENV_PIXELS
from sm64env.load_sm64_CDLL import clear_sm64_exes
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
os.makedirs(f"./data/", exist_ok=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################## TRIPLET MODEL
from triplet_loader import TripletImageLoader, default_image_loader, fast_triplets
from triplet_model import EmbeddingModel
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import PIL

MyTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
])


class TripletModel(EmbeddingModel):
    def __init__(self):
        super().__init__(128)
        self.transform = MyTransform
        self.tripletCriterion = nn.TripletMarginLoss(p=2)
        self.tripletOptimizer = optim.Adam(self.parameters(), lr=0.001)

        # self.dataset = TripletImageLoader('./data', transform=self.transform, gamma=2)
        # self.dataloader = DataLoader(self.dataset, batch_size=1024, shuffle=True, num_workers=16)
    
    def get_rewards(self, game_name, image_save_frequency=30):
        self.dataset = TripletImageLoader('./data', transform=self.transform, gamma=2)
        start = self.dataset.triplet_game_starts[game_name]
        num_triplets = self.dataset.triplet_game_lengths[game_name]
        
        max_anchor = np.max(self.dataset.triplets[start:start+num_triplets, 1]) # largest anchor value in the game
        losses_sum = np.zeros(max_anchor+1)
        triplets_sum = np.zeros(max_anchor+1)
        for triplet_i in range(start, start+num_triplets):
            img1, img2, img3 = self.dataset.__getitem__(triplet_i)
            img1, img2, img3 = img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)

            output1 = self.forward(img1, transform=False)
            output2 = self.forward(img2, transform=False)
            output3 = self.forward(img3, transform=False)
            
            L = self.tripletCriterion(output1, output2, output3)

            anchor_i = self.dataset.triplets[triplet_i][1]
            losses_sum[anchor_i] += L.item()
            triplets_sum[anchor_i] += 1

        avg_losses = losses_sum / triplets_sum
        extended_losses = np.repeat(avg_losses, image_save_frequency)
        return extended_losses

    def forward(self, x, transform=True):
        if transform:
            new_x = [PIL.Image.fromarray(x[i].cpu().numpy().transpose(1, 2, 0)) for i in range(len(x))]
            x = [self.transform(i) for i in new_x]
            x = torch.stack(x)
        return super().forward(x.to(device))

    def train(self, num_batches=10):
        self.dataset = TripletImageLoader('./data', transform=self.transform, gamma=2)
        self.dataloader = DataLoader(self.dataset, batch_size=1024, shuffle=True, num_workers=16)

        avg_loss = 0

        for _ in range(num_batches):
            img1, img2, img3 = next(iter(self.dataloader)) # don't worry, its shuffled
            img1, img2, img3 = img1.unsqueeze(0), img2.unsqueeze(0), img3.unsqueeze(0)

            self.tripletOptimizer.zero_grad()
            
            output1 = self.forward(img1, transform=False)
            output2 = self.forward(img2, transform=False)
            output3 = self.forward(img3, transform=False)

            loss = self.tripletCriterion(output1, output2, output3)
            loss.backward()
            self.tripletOptimizer.step()
            avg_loss += loss.item()
        
        return avg_loss / num_batches




########################################################## TRIPLET MODEL




n_envs = 15
steps_per_iter = 1000
ppo_epochs = 10
mini_batch_size = 4096 # fills ~15GB of VRAM
iter_per_log = 1
iter_per_save = 10

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_env(i):
    def mkenv():
        return  SM64_ENV_PIXELS(server = True, server_port=7777 + i, multi_step=4)
    return mkenv



embedder = TripletModel().to(device)

# agent.load_state_dict(torch.load("ppo_1728480135.3339143_40.pth"))
# agent.load_state_dict(torch.load("ppo_1728480135.3339143_220.pth"))
# agent.load_state_dict(torch.load("ppo_1728491932.1914167_3090.pth"))
# agent.load_state_dict(torch.load("ppo_1728754329.291212_340.pth"))
# agent.load_state_dict(torch.load("ppo_1728804252.0307822_160.pth"))
# agent.load_state_dict(torch.load("ppo_1728814183.1124492_110.pth"))
# agent.load_state_dict(torch.load("ppo_1728877177.6916175_150.pth"))
# agent.load_state_dict(torch.load("models/small_ppo_1734886878.8964658_1100.pth"))
# agent.actor_log_std.data.fill_(0)


run_name = f"triplet_embedder_{time.time()}"


print(torch.cuda.get_device_properties(0).total_memory,torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0))
# print("Parameters: ", sum(p.numel() for p in agent.parameters()))
# print("Size in GB: ", sum(p.numel() for p in agent.parameters()) * 4 / 1024 / 1024 / 1024)
# torch.save(agent.state_dict(), "agent.pth")

envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
# envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])


last_log_time = time.time()
iter = 1
with tqdm.tqdm() as iterbar:
    while True:
        obs, info = envs.reset()
        game_names = [info['game_name'][i] for i in range(n_envs)]
        for _ in tqdm.tqdm(range(steps_per_iter), leave=False):
            # print(torch.cuda.get_device_properties(0).total_memory,torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0))

            # Feed forward
            with torch.no_grad():
                torchObs = embedder.forward(obs)
                dist, value = agent.forward(torchObs)

            # Step
            action_raw = dist.sample()
            stick_actions = clamp_stick(action_raw[:, 0:2] * 80).cpu()

            button_actions = (action_raw[:, 2:5] > 0.8).bool().cpu()
            action = (stick_actions, button_actions)
            next_obs, _, done, truncated, info =  envs.step(action)

            # Calculate storage stuff
            log_prob = dist.log_prob(action_raw)
            entropy += dist.entropy().mean()

            # Storage
            log_probs.append(log_prob)
            values.append(value)
            masks.append(torch.tensor(1 - done, dtype=torch.float32).unsqueeze(1).to(device))
            
            obs_s.append(torchObs) #all the internals of the torchObs are already on the device
            actions.append(action_raw)

            obs = next_obs
            #logging here

        # visualise_curiosity(envs.get_attr('curiosity')[0])

        # Bootstrapping the last obs

        rewards = []
        
        for game_name in game_names:
            game_reward = embedder.get_rewards(game_name)
            print(game_reward)
            game_reward = torch.tensor(game_reward, dtype=torch.float32).to(device)
            rewards.append(game_reward) # add them game-by-game
        
        rewards = torch.stack(rewards).to(device)
        rewards = rewards.transpose(0, 1).unsqueeze(2) # [time, game, 1]

        # Ensure all arrays are the same length as rewards, cutting off the end of the episode
        log_probs = [lp[:len(rewards)] for lp in log_probs]
        values = [v[:len(rewards)] for v in values]
        masks = [m[:len(rewards)] for m in masks]
        obs_s = [obs[:len(rewards)] for obs in obs_s]
        actions = [a[:len(rewards)] for a in actions]

        with torch.no_grad():
            last_torchObs = embedder.forward(obs_s[-1])
            _, last_value = agent.forward(last_torchObs)
            returns = compute_gae(last_value, rewards, masks, values)

        # Cat and detach. Each players' experiences are now on the same dimension
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        obs_s = cat_and_detach_obs(obs_s)
        actions   = torch.cat(actions)
        advantages = returns - values

        ppo_update(ppo_epochs, mini_batch_size, obs_s, actions, log_probs, returns, advantages)

        embedder.train(2000)

        iter += 1
        iterbar.update(1)
        if iter % iter_per_save == 0:
            torch.save(agent.state_dict(), f"models/{run_name}_{iter}.pth")
            torch.save(embedder.state_dict(), f"models/EMBEDDER_{run_name}_{iter}.pth")


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











