# unfortunately you need to run as sudo because I needed the keyboard package
# and the pygame window was corrupting the sm64 window somehow


from sm64env.sm64env_pixels import SM64_ENV_PIXELS
import cv2
import numpy as np
import keyboard
import time
env = SM64_ENV_PIXELS()
obs = env.reset()

# Create window
cv2.namedWindow('SM64 AI DX', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SM64 AI DX', 1000, 1000)

stickX = stickY = buttonA = buttonB = buttonZ = 0
running = True

while running:
    cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        running = False
    elif keyboard.is_pressed('r'):
        obs = env.reset()
    elif keyboard.is_pressed('p'):
        obs = env.step(env.action_space.sample())[0]

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

    buttonA = 1 if keyboard.is_pressed('i') else 0
    buttonB = 1 if keyboard.is_pressed('j') else 0
    buttonZ = 1 if keyboard.is_pressed('o') else 0
    action = [(stickX, stickY), (buttonA, buttonB, buttonZ)]
    obs, reward, done, truncations, info = env.step(action)
    
    # Convert RGB to BGR for OpenCV
    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow('SM64 AI DX', obs_bgr)
    
    if cv2.getWindowProperty('SM64 AI DX', cv2.WND_PROP_VISIBLE) < 1:
        running = False

    time.sleep(0.016)

cv2.destroyAllWindows()