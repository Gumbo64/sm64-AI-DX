import gym.vector
import gym.vector
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from sm64env.load_sm64_CDLL import clear_sm64_exes
from visualiser import visualise_game_tokens, visualise_curiosity
import random
import gym
import time
import pygame
import math
n_envs = 1
multi_step = 4
print_time = 1000

clear_sm64_exes()

pygame.init()
screen = pygame.display.set_mode((400, 300))


env = SM64_ENV_CURIOSITY(multi_step=multi_step, server= True, server_port=7777)



obs, info = env.reset()

stickX = 0
stickY = 0

i = 0
start_time = time.time()
while True:


    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)

    # action = envs.action_space.sample()
    # 
    # obs, reward, done, info = env.step(action)
    visualise_game_tokens(obs)

    
    ### make stickX and stickY from pygame inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                stickY = 80
            elif event.key == pygame.K_s:
                stickY = -80
            elif event.key == pygame.K_a:
                stickX = -80
            elif event.key == pygame.K_d:
                stickX = 80
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_w, pygame.K_s]:
                stickY = 0
            elif event.key in [pygame.K_a, pygame.K_d]:
                stickX = 0


    # stickX, stickY = 0, 80
    # buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
    buttonA = 0
    buttonB = 0
    action = [(stickX, stickY), (buttonA, buttonB, 0)]

    # print(actions)
    obs, reward, done, truncated, info = env.step(action)
    print(info['angle'] / math.pi * 180)
    # print(time.time() - start_time)

    # if i % print_time == 0:
    #     visualise_curiosity(envs.get_attr('curiosity')[0])

    #     print(print_time * multi_step * n_envs / (time.time() - start_time))
    #     i = 0
    #     start_time = time.time()
    i += 1



