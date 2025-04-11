from . import load_sm64_CDLL

import gym
from gym import spaces

import numpy as np

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_NOTHING(gym.Env):
    def __init__(self, multi_step=1, server=True, server_port=7777):
        self.game = load_sm64_CDLL.SM64_GAME(server=server, server_port=server_port)
    
        self.action_space = spaces.Tuple((
            # StickX, StickY
            spaces.Box(low=-80, high=80, shape=(2,), dtype=np.int8),
            # A, B, Z
            spaces.MultiBinary(3),
        ))
                
        self.observation_space = spaces.Sequence(
            spaces.Box(low=0, high=255, shape=(0,), dtype=np.uint8)        
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
        info = self.get_info()

        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        return np.empty((0, 0, 0), dtype=np.uint8)
    
    def get_info(self):
        state = self.game.get_mario_state(0)
        return {
            "pos": np.array(state.pos),
            "vel": np.array(state.vel),
        }

    def calculate_reward(self, obs):
        return 0 # TODO: Implement reward function
    
    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        self.game.set_controller(buttonL=0)
        self.game.step_game(num_steps=20) # takes 20 frames to warp out and in the level
        return self.get_observation(), self.get_info()







