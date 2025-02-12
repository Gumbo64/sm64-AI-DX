# from .load_sm64_CDLL import SM64_GAME, clear_sm64_exes
from . import load_sm64_CDLL

import tqdm

import gym
from gym import spaces

import numpy as np

# bool levelAreaMismatch = ((gNetworkPlayerLocal == NULL)
#     || np->currCourseNum != gNetworkPlayerLocal->currCourseNum
#     || np->currActNum    != gNetworkPlayerLocal->currActNum
#     || np->currLevelNum  != gNetworkPlayerLocal->currLevelNum
#     || np->currAreaIndex != gNetworkPlayerLocal->currAreaIndex);

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_PIXELS(gym.Env):
    def __init__(self, multi_step=1, server=True, server_port=7777):
        self.game = load_sm64_CDLL.SM64_GAME(server=server, server_port=server_port)
    
        self.action_space = spaces.Tuple((
            # StickX, StickY
            spaces.Box(low=-80, high=80, shape=(2,), dtype=np.int8),
            # A, B, Z
            spaces.MultiBinary(3),
        ))
                
        # A variable amount of tokens make up the observation space
        self.observation_space = spaces.Sequence(
            # Each token:
            spaces.Tuple((
                # One-Hot encoding: (Self-Mario / Other-Mario / Point)
                spaces.MultiBinary(3),
                # Position
                spaces.Box(low=-8192, high=8192, shape=(3,), dtype=np.float32),
                # Velocity / Normal 
                spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                # Visits
                spaces.Discrete(1000)
            ))
        )
        self.multi_step = multi_step

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = buttons

        self.game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        self.game.step_game(num_steps=self.multi_step)
        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = False
        truncated = False
        info = {}

        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        return self.game.get_pixels()
    
    def calculate_reward(self, obs):
        return 0 # TODO: Implement reward function
    
    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        return self.get_observation(), {}







