import gym.vector
import gym.vector
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from sm64env.load_sm64_CDLL import clear_sm64_exes
from visualiser import visualise_game_tokens, visualise_curiosity
import random
import gym
import time
n_envs = 16
multi_step = 1
print_time = 1000

clear_sm64_exes()

def make_env(i):
    def mkenv():
        # print(i, i % 16 == 0, 7777 + (i // 16))
        return SM64_ENV_CURIOSITY(multi_step=multi_step, server= (i % 16 == 0), server_port=(7777 + (i // 16)))
        # return SM64_ENV_CURIOSITY(multi_step=multi_step, server=True, server_port=7777 + i)
    return mkenv

envs = gym.vector.AsyncVectorEnv([make_env(i) for i in range(n_envs)], shared_memory=False)
# envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(n_envs)])


obs, info = envs.reset()

i = 0
start_time = time.time()
while True:


    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)

    # action = envs.action_space.sample()
    # 
    # obs, reward, done, info = env.step(action)
    # visualise_game_tokens(obs[0])

    if i % print_time == 0:
        stickX = random.randint(-80, 80)
        stickY = random.randint(-80, 80)
        buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
        action = [(stickX, stickY), (buttonA, buttonB, 0)]
        actions = ([action[0] for _ in range(n_envs)], [action[1] for _ in range(n_envs)])

    # print(actions)
    obs, reward, done, truncated, info = envs.step(actions)
    # print(time.time() - start_time)

    # if i % print_time == 0:
    #     visualise_curiosity(envs.get_attr('curiosity')[0])

    #     print(print_time * multi_step * n_envs / (time.time() - start_time))
    #     i = 0
    #     start_time = time.time()
    i += 1



