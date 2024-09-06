from . import load_sm64_CDLL
from . import curiosity_util
import gym
from gym import spaces
import numpy as np

def isInactive(localPlayer, netPlayer):
    return (netPlayer == None) or (not netPlayer.connected) or (localPlayer.currCourseNum != netPlayer.currCourseNum) or (localPlayer.currActNum != netPlayer.currActNum) or (localPlayer.currLevelNum != netPlayer.currLevelNum) or (localPlayer.currAreaIndex != netPlayer.currAreaIndex)

class SM64_ENV_CURIOSITY(gym.Env):
    def __init__(self, max_visits=1000, server=True, server_port=7777):
        self.game = load_sm64_CDLL.SM64_GAME(server=server, server_port=server_port)
        self.curiosity = curiosity_util.CURIOSITY()
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

        self.max_visits = max_visits

    def step(self, action):
        stick, buttons = action
        
        stickX, stickY = stick
        buttonA, buttonB, buttonZ = buttons

        self.game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        self.game.step_game()
        
        obs = self.get_observation()
        
        reward = self.calculate_reward(obs)
        done = False
        info = {}

        return obs, reward, done, info
    
    
    def get_observation(self, num_players=16, num_points=300):
        player_tokens = []
        
        localNetPlayer = self.game.get_network_player(0)
        for i in range(num_players):
            netPlayer = self.game.get_network_player(i)
            
            if isInactive(localNetPlayer, netPlayer):
                # or netPlayer.name != "AI_BOT"
                continue
            
            state = self.game.get_mario_state(i)

            one_hot = np.array([1,0,0]) if i == 0 else np.array([0,1,0])
            pos = np.array(state.pos)
            vel = np.array(state.vel)
            spacer = np.array([0])

            token = np.concatenate([one_hot, pos, vel, spacer])
            player_tokens.append(token)
        player_tokens = np.array(player_tokens)

        one_hot = np.tile(np.array([0,0,1]), (num_points, 1))
        pos_array, normal_array = self.game.get_raycast_sphere_with_normal(amount=num_points)
        # spacer = np.zeros((num_points,1)) # TODO: Implement visits
        visits = self.curiosity.get_visits_multi(pos_array)
        visits = np.expand_dims(visits, axis=1)

        point_tokens = np.concatenate([one_hot, pos_array, normal_array, visits], axis=1)
        point_tokens = point_tokens[np.where(~np.all(normal_array == 0, axis=1))]

        player_empty = len(player_tokens) == 0
        point_empty = len(point_tokens) == 0
        if player_empty and point_empty:
            return np.zeros((1, 10))
        elif player_empty:
            return point_tokens
        elif point_empty:
            return player_tokens
        return np.concatenate([player_tokens, point_tokens])
    
    def calculate_reward(self, obs):
        tokens = obs
        main_player_indice = np.where(tokens[:, 0] == 1)
        player_token_indices = np.where(tokens[:, 1] == 1)
        # point_token_indices = np.where(tokens[:, 2] == 1)

        my_pos = tokens[main_player_indice][0][3:6]
        other_pos = tokens[player_token_indices][:, 3:6]

        my_visits = self.curiosity.get_visits(my_pos)
        reward = (1 - my_visits / self.max_visits)

        self.curiosity.add_circle(my_pos)
        self.curiosity.add_circles(other_pos)

        return reward 

    def reset(self):
        self.game.set_controller(buttonL=1)
        self.game.step_game()
        return self.get_observation()
    







