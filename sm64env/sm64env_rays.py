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
    def __init__(self, multi_step=4, num_points=1000, fps_amount=100,
                  max_ray_length=8000, server=True, server_port=7777):
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
                # P P P N N N O
                # Position (relative)
                # Normal / Velocity
                # One-Hot encoding: (Self-Mario=1 / Point=0)
            spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        )
        self.multi_step = multi_step
        self.num_points = num_points
        self.max_ray_length = max_ray_length
        self.fps_amount = fps_amount if fps_amount is not None else num_points
        assert self.fps_amount <= self.num_points

        self.my_pos = np.array([0,0,0])
        self.my_vel = np.array([0,0,0])
        self.my_angle = 0

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = [b > 0 for b in buttons]
        
        stickAngle = np.arctan2(stickY, stickX) + math.pi
        # lakituAngle = self.game.get_lakitu_yaw() * np.pi / 0x8000 # convert from sm64 units
        
        newAngle = stickAngle + self.my_angle
        length = min(np.sqrt(stickX**2 + stickY**2), 80)
        stickX = length * np.cos(newAngle)
        stickY = length * np.sin(newAngle)

        self.game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        self.game.step_game(num_steps=self.multi_step)
        
        obs = self.get_observation()
        reward = 0
        done = False
        truncated = False
        info = {}


        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        # PLAYER TOKEN
        state = self.game.get_mario_state(0)
        self.my_pos = np.array(state.pos)
        self.my_vel = np.array(state.vel)

        pos = np.array(state.pos)
        vel = np.array(state.vel)
        one_hot = np.ones((1))

        token = np.concatenate([pos, vel, one_hot])
        player_tokens = np.array([token])

        # POINT TOKENS
        one_hot = np.zeros((self.num_points, 1))
        pos_array, normal_array = self.game.get_raycast_sphere_with_normal(amount=self.num_points, maxRayLength=self.max_ray_length)
        
        # visits = self.curiosity.get_sphere_visits_multi(pos_array)

        visits = np.expand_dims(visits, axis=1)

        point_tokens = np.concatenate([pos_array, normal_array, one_hot], axis=1)
        # point_tokens = point_tokens[np.where(~np.all(normal_array == 0, axis=1))] # Remove zero normals, not necessary anymore

        if self.fps_amount < len(point_tokens):
            point_tokens = point_tokens[fpsample.fps_sampling(point_tokens[:, 0:3], self.fps_amount)]

        # self.avg_visits = 0.9 * self.avg_visits + 0.1 * np.mean(point_tokens[:, 7]) # Rewards are at index 7
        # self.avg_visits = np.mean(point_tokens[:, 7]) # Rewards are at index 7

        
        # distances = np.linalg.norm(point_tokens[:, 0:3] - self.my_pos, axis=1) / self.max_ray_length
        # # distances_square = distances ** 2
        # self.visits_reward = np.mean((point_tokens[:, 7]/self.max_visits) * distances)

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
        angle = np.array(state.faceAngle) # 3d array
        angle = angle[1] * np.pi / 0x8000 # convert from sm64 units
        self.my_angle = angle


        rotation_matrix = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
        tokens[:, 0:3] = np.dot(tokens[:, 0:3], rotation_matrix)
        tokens[:, 3:6] = np.dot(tokens[:, 3:6], rotation_matrix)

        tokens[:, 0:3] /= self.max_ray_length # Normalize position
        return tokens

    def reset(self):
        self.avg_visits = 0
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        if self.soft_reset:
            self.curiosity.soft_reset()
        else:
            self.curiosity.reset()
        return self.get_observation(), {}
    







