# unfortunately you need to run as sudo because I needed the keyboard package
# and the pygame window was corrupting the sm64 window somehow


from sm64env.sm64env_pixels import SM64_ENV_PIXELS
import cv2
import csv
import numpy as np
import keyboard
import uuid
import time
from PIL import Image
import os
import random
# test that the keyboard perms are working
keyboard.is_pressed('q')


img_save_frequency = 10
max_game_length = 1000

os.makedirs(f"./data/", exist_ok=True)

env = SM64_ENV_PIXELS()
obs = env.reset()

# Create window
cv2.namedWindow('SM64 AI DX', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SM64 AI DX', 1000, 1000)

stickX = stickY = buttonA = buttonB = buttonZ = 0

def new_run(old_file=None):
    if old_file:
        old_file.close()
    my_id = str(uuid.uuid4()).replace("-", "").replace(" ", "")
    os.makedirs(f"./data/{my_id}", exist_ok=True)

    file = open(f"./data/{my_id}/actions.csv", mode='w', newline='')
    writer = csv.writer(file)
    return my_id, 0, file, writer

run_id, run_time_counter, run_csv_file, run_csv_writer = new_run()
resetted_last = False

running = True

while running:
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        running = False
    if keyboard.is_pressed('r') or run_time_counter >= max_game_length:
        if not resetted_last:
            obs = env.reset()
            run_id, run_time_counter, run_csv_file, run_csv_writer = new_run(run_csv_file)
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
    cv2.imshow('SM64 AI DX', obs_bgr)
    if run_time_counter % img_save_frequency == 0:
        save_path = f"./data/{run_id}/{run_time_counter // img_save_frequency}.png"
        cv2.imwrite(save_path, obs_bgr)
    run_csv_writer.writerow([stickX, stickY, buttonA, buttonB, buttonZ])
    run_time_counter += 1
    
    if cv2.getWindowProperty('SM64 AI DX', cv2.WND_PROP_VISIBLE) < 1:
        running = False

    time.sleep(0.016)

cv2.destroyAllWindows()