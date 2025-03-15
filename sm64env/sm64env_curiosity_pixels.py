from . import load_sm64_CDLL
from . import curiosity_util
import gym
from gym import spaces
import numpy as np
import fpsample
import math

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_CURIOSITY(gym.Env):
    def __init__(self, multi_step=4, max_visits=500, num_points=1000, fps_amount=100, soft_reset=False,
                  max_ray_length=8000, server=True, server_port=7777):
        self.game = load_sm64_CDLL.SM64_GAME(server=server, server_port=server_port)
        self.curiosity = curiosity_util.CURIOSITY(max_visits=max_visits)
    
        self.action_space = spaces.Tuple((
            # StickX, StickY
            spaces.Box(low=-80, high=80, shape=(2,), dtype=np.int8),
            # A, B, Z
            spaces.MultiBinary(3),
        ))
                
        # A variable amount of tokens make up the observation space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(72, 128, 3), dtype=np.uint8
        )
        
        self.multi_step = multi_step
        self.max_visits = max_visits
        self.num_points = num_points
        self.max_ray_length = max_ray_length
        self.fps_amount = fps_amount if fps_amount is not None else num_points
        assert self.fps_amount <= self.num_points
        self.soft_reset = soft_reset

        self.my_pos = np.array([0,0,0])
        self.my_vel = np.array([0,0,0])
        self.my_angle = 0
        self.avg_visits = 0

        self.vel_reward = 0
        self.curiosity_reward = 0

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = [b > 0 for b in buttons]
        
        # normalise stick
        newAngle = np.arctan2(stickY, stickX)

        length = min(np.sqrt(stickX**2 + stickY**2), 79)
        stickX = length * np.cos(newAngle)
        stickY = length * np.sin(newAngle)

        self.game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        self.game.step_game(num_steps=self.multi_step)
        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = False
        truncated = False
        info = {
            "curiosity_reward": self.curiosity_reward,
            "vel_reward": self.vel_reward,
            "avg_visits": self.avg_visits
        }


        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        # PLAYER TOKEN
        state = self.game.get_mario_state(0)
        self.my_pos = np.array(state.pos)
        self.curiosity.add_circle(self.my_pos)
        img = self.game.get_pixels()
        if img.shape[0] == 0:
            img = np.zeros((72, 128, 3), dtype=np.float32)
        return img

    def calculate_reward(self, obs):
        # Use my_pos to calculate reward
        if len(self.my_pos) == 0:
            return 0

        self.visits = self.curiosity.get_sphere_visits(self.my_pos)
        self.curiosity_reward = np.exp(- 4 * self.visits / self.max_visits)
        self.curiosity_reward = np.clip(np.mean(self.curiosity_reward), 0, 1)


        self.vel_reward = np.clip(math.sqrt(self.my_vel[0] ** 2 + self.my_vel[2] ** 2) / 50, 0, 1)
        
        reward = 0.9 * self.curiosity_reward + 0.1 * self.vel_reward
        # reward = 0.5 * self.curiosity_reward + 0.5 * self.vel_reward
        # reward = 0.75 * curiosity_reward + 0.25 * vel_reward
        return reward 

    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        self.game.set_controller(buttonL=0)
        self.game.step_game(num_steps=20) # takes 20 frames to warp out and in the level

        self.avg_visits = 0
        if self.soft_reset:
            self.curiosity.soft_reset()
        else:
            self.curiosity.reset()

        return self.get_observation(), self.get_info()







