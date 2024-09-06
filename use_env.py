from sm64env.sm64env_example import SM64_ENV
from visualiser import visualise_game_tokens
import random
env = SM64_ENV()

obs = env.reset()

while True:
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
    buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
    action = [(stickX, stickY), (buttonA, buttonB, 0)]
    # action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    visualise_game_tokens(obs)

