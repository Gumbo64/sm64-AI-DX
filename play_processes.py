import gym.vector
import gym.vector
from sm64env.sm64env_example import SM64_ENV
from visualiser import visualise_game_tokens, visualise_curiosity
import random
import gym
import time
n_envs = 16

def make_env(i):
    def mkenv():
        return SM64_ENV(server=(i % 16 == 0), server_port=(7777 + (i // 16)))
    return mkenv

envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
# envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])


obs, info = envs.reset()

while True:
    start_time = time.time()
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
    buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
    action = [(stickX, stickY), (buttonA, buttonB, 0)]
    # action = envs.action_space.sample()
    # 
    # obs, reward, done, info = env.step(action)
    # visualise_game_tokens(obs)
    visualise_curiosity(envs.get_attr('curiosity')[0])
    
    actions = ([action[0] for _ in range(n_envs)], [action[1] for _ in range(n_envs)])

    # print(actions)
    envs.step(actions)
    print(time.time() - start_time)


