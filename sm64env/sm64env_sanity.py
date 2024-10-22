from . import load_sm64_CDLL
from . import curiosity_util
import gym
from gym import spaces
import numpy as np
import fpsample
import math

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_SANITY(gym.Env):
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
                # One-Hot encoding: (Self-Mario / Other-Mario / Point)
                # Relative Position
                # Velocity / Normal
                # Face Vector
            spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        )
        self.multi_step = multi_step
        self.num_points = num_points
        self.max_ray_length = max_ray_length
        self.fps_amount = fps_amount if fps_amount is not None else num_points

        self.my_pos = np.array([0,0,0])
        self.goal_pos = np.array([5524.7285, 3072, 1187.4833])
        self.my_vel = np.array([0,0,0])

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = buttons
        
        stickAngle = np.atan2(stickX, stickY)

        lakituAngle = self.game.get_lakitu_yaw() * np.pi / 0x8000 # convert from sm64 units
        newAngle = stickAngle - lakituAngle
        length = min(np.linalg.norm(np.array([stickX, stickY])), 79)
        stickX = length * np.cos(newAngle)
        stickY = length * np.sin(newAngle)



        # print(stickX, stickY)


        self.game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        self.game.step_game(num_steps=self.multi_step)
        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = False
        truncated = False
        info = {}


        return obs, reward, done, truncated, info
    
    
    def get_observation(self):
        # Goal token
        goal_token = np.array([[0,1,0] + list(self.goal_pos) + [0,0,0] + [0,0,0]])


        player_tokens = []
                
        state = self.game.get_mario_state(0)
        one_hot = np.array([1,0,0])
        self.my_pos = np.array(state.pos)
        self.my_vel = np.array(state.vel)
        pos = np.array(state.pos)
        vel = np.array(state.vel)
        faceAngle = np.array(state.faceAngle) # 3d array
        faceAngle = faceAngle[1] * np.pi / 0x8000 # convert from sm64 units
        faceVector = np.array([math.sin(faceAngle), 0, math.cos(faceAngle)])

        token = np.concatenate([one_hot, pos, vel, faceVector])
        player_tokens.append(token)
        player_tokens = np.array(player_tokens)



        one_hot = np.tile(np.array([0,0,1]), (self.num_points, 1))
        pos_array, normal_array = self.game.get_raycast_sphere_with_normal(amount=self.num_points, maxRayLength=self.max_ray_length)
        
        filler = np.zeros((self.num_points, 3))

        point_tokens = np.concatenate([one_hot, pos_array, normal_array, filler], axis=1)
        point_tokens = point_tokens[np.where(~np.all(normal_array == 0, axis=1))] # Remove zero normals

        if self.fps_amount < len(point_tokens):
            point_tokens = point_tokens[fpsample.fps_sampling(point_tokens[:, 3:6], self.fps_amount)]


        player_empty = len(player_tokens) == 0
        point_empty = len(point_tokens) == 0
        if player_empty and point_empty:
            tokens = np.zeros((1, 12))
        elif player_empty:
            tokens = point_tokens
        elif point_empty:
            player_tokens[:, 6:9] /= 50 # Normalize velocity for players (not point normals though)
            tokens = player_tokens
        else:
            player_tokens[:, 6:9] /= 50 # Normalize velocity for players (not point normals though)
            tokens = np.concatenate([player_tokens, point_tokens])
 
        tokens = np.concatenate([tokens, goal_token])
        
        if len(self.my_pos) == 0:
            return tokens

        # Rotate and position the coordinates to the lakitu's perspective
        # origin = self.game.get_lakitu_pos()
        # diff = self.my_pos - origin

        # tokens[:, 3:6] -= origin # Translate position
        
        # angle = math.atan2(diff[2], diff[0])

        # rotation_matrix = np.array([[math.cos(angle), 0, -math.sin(angle)], [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
        # tokens[:, 3:6] = np.dot(tokens[:, 3:6], rotation_matrix)
        # tokens[:, 6:9] = np.dot(tokens[:, 6:9], rotation_matrix)
        # tokens[: ,9:12] = np.dot(tokens[:, 9:12], rotation_matrix)

        # tokens[:, 3:6] /= self.max_ray_length # Normalize position
        tokens[:, 3:6] /= 8192
        return tokens

    def calculate_reward(self, obs):
        # Use my_pos to calculate reward
        if len(self.my_pos) == 0:
            return 0
        pos_reward = 1-np.linalg.norm(self.my_pos - self.goal_pos) / (2*8192)
        return pos_reward

    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        return self.get_observation(), {}
    







