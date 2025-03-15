import matplotlib.pyplot as plt
import torch as torch

from triplet_loader import default_image_loader
from triplet_model import EmbeddingModel, MyTransform
import os
from tqdm import tqdm
import numpy as np

# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel().to(device)
for param in model.parameters():
    param.requires_grad = False
transform = MyTransform

# Load the trained model
model.load_state_dict(torch.load('triplet.pth'))
model.eval()


files = []
for game_name in os.listdir('./data'):
    for file in os.listdir(os.path.join('./data', game_name)):
        if file.endswith('.png'):
            files.append(os.path.join('./data', game_name, file))

# pairs = []
embeddings = []
images = []
with torch.no_grad():
    for file in tqdm(files):
        ending = file.split('/')[-1]
        number = int(ending.split('.')[0])

        img = default_image_loader(os.path.join(file))
        img = transform(img).unsqueeze(0).to(device)
        embedding = model(img).cpu().numpy()[0]
        # pairs.append((img, embedding))
        embeddings.append(embedding)
        images.append(img.cpu().numpy())

embeddings = np.array(embeddings)


# playable game with the window
# unfortunately you need to run as sudo because I needed the keyboard package
# and the pygame window was corrupting the sm64 window somehow


from sm64env.sm64env_pixels import SM64_ENV_PIXELS
import cv2
import numpy as np
import keyboard
import time
from PIL import Image
import os
import random
from sklearn.metrics.pairwise import euclidean_distances

# test that the keyboard perms are working
keyboard.is_pressed('q')


img_save_frequency = 10

os.makedirs(f"./data/", exist_ok=True)

env = SM64_ENV_PIXELS()
obs = env.reset()


stickX = stickY = buttonA = buttonB = buttonZ = 0

resetted_last = False
running = True

while running:
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        running = False
    if keyboard.is_pressed('r'):
        if not resetted_last:
            obs = env.reset()
            resetted_last = True
    else:
        resetted_last = False

    if keyboard.is_pressed('w'):
        stickY = 80
    elif keyboard.is_pressed('s'):
        stickY = -80
    else:
        stickY = 0

    if keyboard.is_pressed('a'):
        stickX = -80
    elif keyboard.is_pressed('d'):
        stickX = 80
    else:
        stickX = 0

    if keyboard.is_pressed('p'):
        stickX = random.randint(-80, 80)
        stickY = random.randint(-80, 80)
        buttonA, buttonB = random.choices([0, 1], weights=[0.99, 0.01], k=2)

    buttonA = 1 if keyboard.is_pressed('i') else 0
    buttonB = 1 if keyboard.is_pressed('j') else 0
    buttonZ = 1 if keyboard.is_pressed('o') else 0
    action = [(stickX, stickY), (buttonA, buttonB, buttonZ)]
    obs, reward, done, truncations, info = env.step(action)
    

    # Convert RGB to BGR for OpenCV
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    # cv2.imshow('SM64 AI DX', obs_bgr)

    # find the closest image according to the embeddings
    obs_img = Image.fromarray(obs)
    obs_img = transform(obs_img).unsqueeze(0).to(device)
    embedding = model(obs_img).cpu().detach().numpy()[0]

    distances = euclidean_distances(embeddings, [embedding])
    closest_index = np.argmin(distances)
    print(closest_index)
    # closest_img = images[closest_index]
    closest_img = cv2.imread(files[closest_index])
    # closest_img = cv2.cvtColor(closest_img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Closest Image', closest_img)
    
    
    # closest_img = Image.open(pairs[closest][0])
    # closest_img = np.array(closest_img)
    # closest_img = cv2.cvtColor(closest_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Closest Image', closest_img)

    
    time.sleep(0.016)

cv2.destroyAllWindows()