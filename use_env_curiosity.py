from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from visualiser import visualise_game_tokens, visualise_curiosity
import random
import time
env = SM64_ENV_CURIOSITY()
obs = env.reset()

i = 1
start_time = time.time()
while True:
    
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
    buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
    action = [(stickX, stickY), (buttonA, buttonB, 0)]
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # visualise_game_tokens(obs)
    # visualise_curiosity(env.curiosity)


