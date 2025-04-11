from sm64env.sm64env_nothing import SM64_ENV_NOTHING
# from sm64env.sm64env_pixels import SM64_ENV_PIXELS
from sm64env.load_sm64_CDLL import clear_sm64_exes
from sm64env.dfs_util import DFS_UTIL
from visualiser import visualise_dfs_time
import random
import pickle
import gym
import numpy as np
import tqdm
import time
n_envs = 8
clear_sm64_exes()
def make_env(i):
    def mkenv():
        return SM64_ENV_NOTHING(multi_step=1, server=True, server_port=(7777 + i))
    return mkenv

def fill_positions(infos):
    p = np.zeros((n_envs, 3), dtype=float)
    for i in range(n_envs):
        p[i] = infos["pos"][i]
    return p

# print("Parameters: ", sum(p.numel() for p in agent.parameters()))
# print("Size in GB: ", sum(p.numel() for p in agent.parameters()) * 4 / 1024 / 1024 / 1024)
# torch.save(agent.state_dict(), "agent.pth")

# env = SM64_ENV_NOTHING(multi_step=4, server=True, server_port=7777)
# obs, info = env.reset()

envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
obss, infos = envs.reset()
dfs = DFS_UTIL()
# with open('dfs.pickle', 'rb') as file:
#     dfs = pickle.load(file)





# prev_position = np.array(info["pos"])

prev_positions = None
positions = fill_positions(infos)
dfs.add_circles(positions, 0, old_centres=prev_positions)

time_max = 1600
id = 0
with tqdm.tqdm() as pbar:
    while True:
        for timestamp in tqdm.tqdm(range(1,time_max), leave=False):
            # actions_raw = np.random.rand(n_envs, 5) * 2 - 1

            # actions_stick = actions_raw[:, 0:2] * 80
            # actions_buttons = actions_raw[:, 2:5] > 0.95
            # actions = (actions_stick.astype(np.int8), actions_buttons.astype(np.int8))
            # print(actions)

            actions = [
                [(random.randint(-80, 80), random.randint(-80, 80)) for _ in range(n_envs)],
                [(0, 0, 0) for _ in range(n_envs)]
            ]
            # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
            # buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
            # action = [(stickX, stickY), (buttonA, buttonB, 0)]

            obss, rewards, dones, truncations, infos = envs.step(actions)
            # obs, rewards, dones, truncations, infos = env.step(action)

            positions = fill_positions(infos)
            # position = np.array(info["pos"])

            dfs.add_circles(positions, timestamp, old_centres=prev_positions)
            prev_positions = positions.copy()

            # dfs.add_circle(position, timestamp, old_centre=prev_position)
            # prev_position = position.copy()

            # time.sleep(0.16)

        with open('dfs.pickle', 'wb') as file:
            pickle.dump(dfs, file)

        # visualise_dfs_time(dfs, time_max)
        envs.reset()
        # env.reset()
        pbar.update(1)


    

    # visualise_game_tokens(obs)

