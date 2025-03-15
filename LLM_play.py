from ollama import chat
from ollama import ChatResponse

from sm64env import sm64env_pixels
from PIL import Image
import numpy as np

image_address = "game.png"
# model = "hf.co/unsloth/gemma-3-27b-it-GGUF:Q4_K_M"
# model = "gemma3:27b"
model = "gemma3:12b"

env = sm64env_pixels.SM64_ENV_PIXELS(server=True, server_port=7777, multi_step=4)

obs, info = env.reset()

while True:

    # promptString = f"""
    # Give me your suggested button presses and joystick direction for the given screenshot. Explore the level to the best of your abilities. Reminder that positive stickX goes to the right, and positive stickY points it up. Give these inputs in the format:
    # STARTING_INPUT
    # StickX =  _(number from -100 to 100)_
    # StickY = _(number from -100 to 100)_
    # A button = _(true or false)_
    # B button = _(true or false)_
    # Z button = _(true or false)_
    # """

    promptString = f"""
    Give me a suggested action for the given screenshot. Explore the level to the best of your abilities. Choose any amount of the following actions or none at all. Give these inputs in the format:
    FORWARD
    BACKWARD
    LEFT
    RIGHT
    JUMP
    CROUCH
    """

    # Convert the observation to an image and save it
    image = Image.fromarray(obs)
    image.save(image_address)
    while True:
        try:
            response: ChatResponse = chat(model=model, messages=[
            {
                'role': 'user',
                'content': promptString,
                'images': [image_address]
            },
            ])
            s = response.message.content
            # print(s)
            inputs = {}
            inputs['stickX'] = 0
            inputs['stickY'] = 0
            inputs['buttonA'] = False
            inputs['buttonB'] = False
            inputs['buttonZ'] = False

            if "FORWARD" in s:
                inputs['stickY'] = 100
            if "BACKWARD" in s:
                inputs['stickY'] = -100
            if "LEFT" in s:
                inputs['stickX'] = -100
            if "RIGHT" in s:
                inputs['stickX'] = 100
            if "JUMP" in s:
                inputs['buttonA'] = True
            if "KICK" in s:
                inputs['buttonB'] = True
            if "CROUCH" in s:
                inputs['buttonZ'] = True
            break
        except:
            print("FAIL PARSING")

            
    # Apply sigmoid function and scale for stickX and stickY

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    inputs['stickX'] = int(sigmoid(inputs['stickX']) * 80)
    inputs['stickY'] = int(sigmoid(inputs['stickY']) * 80)

    action = [inputs['stickX'], inputs['stickY']], [inputs['buttonA'], inputs['buttonB'], inputs['buttonZ']]

    obs, reward, done, truncated, info = env.step(action)
