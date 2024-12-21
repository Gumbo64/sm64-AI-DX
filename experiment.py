from sm64env.sm64env_example import SM64_ENV
from sm64env.sm64env_curiosity import SM64_ENV_CURIOSITY
from visualiser import visualise_game_tokens
import random
import numpy as np

# env = SM64_ENV_CURIOSITY()

# obs = env.reset()

# class node():
#   def __init__(self):
#     self.left = None
#     self.right = None
#     self.action = None

depth = 5
degree = 2

size = degree ** depth + 1

tree = np.random.randint(-80, 80, (size, 5))

print(tree)


# def make_tree(depth):
#   tree = node()
#   if (depth > 1):
#     tree.left = make_tree(depth-1)
#     tree.right = make_tree(depth-1)

#   stickX = random.randint(-80, 80)
#   stickY = random.randint(-80, 80)
#   # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
#   buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
#   tree.action = [(stickX, stickY), (buttonA, buttonB, 0)]
#   return tree


# def fill_tree(tree, depth):
#   if (not tree):
#     return make_tree(depth)
  
#   if (depth > 1):
#     tree.left = fill_tree(tree.left, depth-1)
#     tree.right = fill_tree(tree.right, depth-1)
  
#   return tree

# def print_tree(tree):
#   print(tree.action)

#   print('l')
#   if (tree.left):
#     print_tree(tree.left)

#   print('r')
#   if (tree.right):
#     print_tree(tree.right)

# tree = make_tree(15)
# print("-------------------------")
# print_tree(tree)

# tree.right = None
# print("-------------------------")
# print_tree(tree)

# tree = fill_tree(tree, 15)
# print("-------------------------")
# print_tree(tree)

# while True:
#   stickX = random.randint(-80, 80)
#   stickY = random.randint(-80, 80)
#   # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.99, 0.01], k=3)
#   buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)
#   action = [(stickX, stickY), (buttonA, buttonB, 0)]
#   # action = env.action_space.sample()
#   obs, reward, done, truncations, info = env.step(action)
#   # visualise_game_tokens(obs)

