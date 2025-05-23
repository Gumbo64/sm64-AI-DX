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
    def __init__(self, multi_step=4, max_visits=500, num_points=100, fps_amount=None, soft_reset=False,
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
        self.observation_space = spaces.Sequence(
            # Each token:
                # P P P N N N O V
                # Position (relative)
                # Normal / Velocity
                # One-Hot encoding: (Self-Mario=1 / Point=0)
                # Visits at the given position
            spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
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


        self.game.step_game(num_steps=self.multi_step, stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)

        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = False
        truncated = False
        info = {
            "curiosity_reward": self.curiosity_reward,
            "vel_reward": self.vel_reward,
            "avg_visits": self.avg_visits
        }
        info = {**info, **self.get_info()}

        return obs, reward, done, truncated, info
    
    def get_info(self):
        state = self.game.get_mario_state(0)

        camera_pos = self.game.get_lakitu_pos()
        dir = state.pos - camera_pos
        angle = math.atan2(dir[2], dir[0])

        return {
            "pos": np.array(state.pos),
            "vel": np.array(state.vel),
            "angle": angle,
        }
    
    def get_observation(self):
        # PLAYER TOKEN
        state = self.game.get_mario_state(0)
        self.my_pos = np.array(state.pos)
        self.my_vel = np.array(state.vel)

        pos = np.array(state.pos)
        vel = np.array(state.vel)
        one_hot = np.ones((1))
        visits = np.array([self.curiosity.get_visits(pos)])

        token = np.concatenate([pos, vel, one_hot, visits])
        player_tokens = np.array([token])

        # POINT TOKENS
        one_hot = np.zeros((self.num_points, 1))
        pos_array, normal_array = self.game.get_raycast_sphere_with_normal(amount=self.num_points, maxRayLength=self.max_ray_length)
        
        visits = self.curiosity.get_visits_multi(pos_array)

        # visits = self.curiosity.get_sphere_visits_multi(pos_array)

        visits = np.expand_dims(visits, axis=1)

        point_tokens = np.concatenate([pos_array, normal_array, one_hot, visits], axis=1)
        # point_tokens = point_tokens[np.where(~np.all(normal_array == 0, axis=1))] # Remove zero normals, not necessary anymore

        if self.fps_amount < len(point_tokens):
            point_tokens = point_tokens[fpsample.fps_sampling(point_tokens[:, 0:3], self.fps_amount)]

        # self.avg_visits = 0.9 * self.avg_visits + 0.1 * np.mean(point_tokens[:, 7]) # Rewards are at index 7
        # self.avg_visits = np.mean(point_tokens[:, 7]) # Rewards are at index 7

        
        # distances = np.linalg.norm(point_tokens[:, 0:3] - self.my_pos, axis=1) / self.max_ray_length
        # # distances_square = distances ** 2
        # self.visits_reward = np.mean((point_tokens[:, 7]/self.max_visits) * distances)

        self.visits = point_tokens[:, 7]
        self.distances = np.linalg.norm(point_tokens[:, 0:3] - self.my_pos, axis=1)

        # print(self.avg_visits)
        self.curiosity.add_circles(point_tokens[:, 0:3]) # Curiosity update for each point

        player_tokens[:, 3:6] /= 50 # Normalize velocity for players (not point normals though)
        tokens = np.concatenate([player_tokens, point_tokens])

        if len(self.my_pos) == 0:
            return tokens

        # Rotate and position the coordinates to the mario's perspective
        origin = self.my_pos

        tokens[:, 0:3] -= origin # Translate position
        
        # angle = math.atan2(dir[2], dir[0])

        camera_pos = self.game.get_lakitu_pos()
        dir = camera_pos - origin
        angle = math.atan2(dir[2], dir[0])

        rotation_matrix = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
        tokens[:, 0:3] = np.dot(tokens[:, 0:3], rotation_matrix)
        tokens[:, 3:6] = np.dot(tokens[:, 3:6], rotation_matrix)

        tokens[:, 0:3] /= self.max_ray_length # Normalize position
        tokens[:, 7] /= self.max_visits # Normalize visits
        return tokens

    def calculate_reward(self, obs):
        # Use my_pos to calculate reward
        if len(self.my_pos) == 0:
            return 0

        # self.curiosity_reward = 1 - np.mean(self.visits / self.max_visits)
        # self.curiosity_reward = np.exp(- 4 * np.mean(self.visits / self.max_visits))

        self.curiosity_reward = np.mean(np.exp(- 4 * self.visits / self.max_visits))

        self.curiosity_reward = np.clip(self.curiosity_reward, 0, 1)


        self.vel_reward = np.clip(math.sqrt(self.my_vel[0] ** 2 + self.my_vel[2] ** 2) / 50, 0, 1)
        # self.vel_reward = np.clip(np.linalg.norm(self.my_vel) / 50, 0, 1)
        
        # reward = 0.9 * self.curiosity_reward + 0.1 * self.vel_reward
        reward = 0.5 * self.curiosity_reward + 0.5 * self.vel_reward
        # reward = 0.75 * curiosity_reward + 0.25 * vel_reward
        return reward 

    def reset(self):
        self.avg_visits = 0
        self.game.step_game(buttonL=1)
        if self.soft_reset:
            self.curiosity.soft_reset()
        else:
            self.curiosity.reset()
        return self.get_observation(), {}
    







