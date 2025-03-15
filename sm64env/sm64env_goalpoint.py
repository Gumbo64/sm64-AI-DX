from . import load_sm64_CDLL
from . import curiosity_util
import gym
from gym import spaces
import numpy as np
import fpsample

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_GOALPOINT(gym.Env):
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
                # Spacer
            spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        )
        self.multi_step = multi_step
        self.num_points = num_points
        self.max_ray_length = max_ray_length
        self.fps_amount = fps_amount if fps_amount is not None else num_points

        self.my_pos = []

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
        player_tokens = []
        self.my_pos = []
        localNetPlayer = self.game.get_network_player(0)
        for i in range(16):
            netPlayer = self.game.get_network_player(i)
            
            if isInactive(localNetPlayer, netPlayer):
                # or netPlayer.name != "AI_BOT"
                continue
            
            state = self.game.get_mario_state(i)

            if i == 0:
                one_hot = np.array([1,0,0])
                self.my_pos = np.array(state.pos)
            else:
                one_hot = np.array([0,1,0])

            pos = np.array(state.pos)
            vel = np.array(state.vel)
            spacer = np.array([0])

            token = np.concatenate([one_hot, pos, vel, spacer])
            player_tokens.append(token)
        player_tokens = np.array(player_tokens)

        one_hot = np.tile(np.array([0,0,1]), (self.num_points, 1))
        pos_array, normal_array = self.game.get_raycast_sphere_with_normal(amount=self.num_points, maxRayLength=self.max_ray_length)
        spacer = np.array([0])
        spacer = np.expand_dims(spacer, axis=1)

        point_tokens = np.concatenate([one_hot, pos_array, normal_array, spacer], axis=1)
        point_tokens = point_tokens[np.where(~np.all(normal_array == 0, axis=1))] # Remove zero normals

        if self.fps_amount < len(point_tokens):
            point_tokens = point_tokens[fpsample.fps_sampling(point_tokens[:, 1:4], self.fps_amount)]


        player_empty = len(player_tokens) == 0
        point_empty = len(point_tokens) == 0
        if player_empty and point_empty:
            tokens = np.zeros((1, 10))
        elif player_empty:
            tokens = point_tokens
        elif point_empty:
            tokens = player_tokens
        else:
            tokens = np.concatenate([player_tokens, point_tokens])

        if len(self.my_pos) == 0:
            return tokens

        # Make coordinates relative to the main player
        my_x, my_y, my_z = self.my_pos
        tokens[:] -= np.array([0,0,0,my_x,my_y,my_z,0,0,0,0])
        tokens[:, 3:6] /= 8192 # Normalize position
        tokens[:, 6:9] /= 50    # Normalize velocity
        return tokens

    def calculate_reward(self, obs):
        # Use my_pos to calculate reward
        if len(self.my_pos) == 0:
            return 0

        goalPos = np.array([5524.7285, 3072, 1187.4833])
        dist = goalPos - self.my_pos
        d = np.linalg.norm(dist)
        return d/8192

    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        self.curiosity.reset()
        return self.get_observation(), {}
    







