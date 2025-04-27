# from .load_sm64_CDLL import SM64_GAME, clear_sm64_exes
from . import load_sm64_CDLL

import tqdm

import gym
from gym import spaces

import numpy as np

import uuid
import os

import cv2

# bool levelAreaMismatch = ((gNetworkPlayerLocal == NULL)
#     || np->currCourseNum != gNetworkPlayerLocal->currCourseNum
#     || np->currActNum    != gNetworkPlayerLocal->currActNum
#     || np->currLevelNum  != gNetworkPlayerLocal->currLevelNum
#     || np->currAreaIndex != gNetworkPlayerLocal->currAreaIndex);

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_PIXELS(gym.Env):
    def __init__(self, image_save_frequency=30, multi_step=1, server=True, server_port=7777):
        self.game = load_sm64_CDLL.SM64_GAME(server=server, server_port=server_port)
    
        self.action_space = spaces.Tuple((
            # StickX, StickY
            spaces.Box(low=-80, high=80, shape=(2,), dtype=np.int8),
            # A, B, Z
            spaces.MultiBinary(3),
        ))
                
        # A variable amount of tokens make up the observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(144, 256, 3), dtype=np.uint8
        )
        
        self.multi_step = multi_step
        self.image_save_frequency = image_save_frequency

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = buttons

        self.game.step_game(num_steps=self.multi_step, stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)

        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = False
        truncated = False
        info = {
            "game_name": self.game_name
        }

        # Save image
        if self.run_time_counter % self.image_save_frequency == 1:
            save_path = os.path.join("./data", self.game_name, f"{self.run_time_counter // self.image_save_frequency}.png")
            obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, obs_bgr)
        self.run_time_counter += 1

        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        return self.game.get_pixels()
    
    def calculate_reward(self, obs):
        return 0 # TODO: Implement reward function
    
    def reset(self):
        self.run_time_counter = 0
        self.game_name = str(uuid.uuid4()).replace("-", "").replace(" ", "")
        os.makedirs(f"./data/{self.game_name}", exist_ok=True)
        
        self.game.step_game(buttonL=1)

        obs, _, _, _, info = self.step((np.array([0,0]), np.array([0,0,0])))

        return obs, info







