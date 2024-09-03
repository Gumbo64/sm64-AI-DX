from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time
import ctypes
clear_sm64_exes()

game = SM64_GAME(server=True, server_port=7777)


# with tqdm.tqdm() as pbar:
while True:
    stickX = random.randint(-80, 80)
    stickY = random.randint(-80, 80)
    buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
    # buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]
    game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
    game.step_game()
    marioState = game.get_mario_state(1)
    networkPlayer = game.get_network_player(1)
    
    # print([int(marioState.pos[i]) for i in range(3)])
