from sm64env.sm64env_example import SM64_ENV
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from visualiser import visualise_game_tokens
import random

from transformerModel import SetTransformer
import torch


def clip(x, lo, hi):
    return max(lo, min(hi, x))

env = SM64_ENV_CURIOSITY()


# num heads has to be a factor of the input dimension for some reason idk man
input_dim = 64 # Dimension of each token
num_heads = 4 # Number of heads in multi-head attention


# Set Transformer
agent = SetTransformer(input_dim, num_heads)
print("Number of parameters in the agent:", sum(p.numel() for p in agent.parameters()))

obs, info = env.reset()

obsTensor = torch.tensor(obs, dtype=torch.float32)

while True:
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
    # buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
    # action = [(stickX, stickY), (buttonA, buttonB, 0)]
    # action = env.action_space.sample()

    obsTensorBatched = obsTensor.unsqueeze(0)
    paddedObsTensor = torch.nn.functional.pad(obsTensorBatched, (0, input_dim - obsTensorBatched.size(2)))
    agentReturn = agent(paddedObsTensor)[0]
    print(agentReturn)
    stickX = clip(int(agentReturn[0] * 80), -80, 80)
    stickY = clip(int(agentReturn[1] * 80), -80, 80)
    buttonA = agentReturn[2] > 0
    buttonB = agentReturn[3] > 0
    buttonZ = agentReturn[4] > 0
    action = [(stickX, stickY), (buttonA, buttonB, buttonZ)]
    obs, reward, done, truncations, info = env.step(action)
    obsTensor = torch.tensor(obs, dtype=torch.float32)
    # visualise_game_tokens(obs)

