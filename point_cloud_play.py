from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
import tqdm
import random
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt

clear_sm64_exes()

game = SM64_GAME(server=True, server_port=7777)
points_array = np.array([[0,0,0]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# with tqdm.tqdm() as pbar:
while True:

    # stickX = random.randint(-80, 80)
    # stickY = random.randint(-80, 80)
    # buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
    stickX, stickY = 80, 60
    buttonA, buttonB, buttonZ = 0, 0, 0
    buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]
    game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
    game.step_game()
    marioState = game.get_mario_state(0)
    networkPlayer = game.get_network_player(0)

    start_time = time.time()

    
    # pos = np.array(marioState.pos)
    # dir = np.array([0, -10000, 0])
    # for i in range(3000):
        
    #     new_point = game.get_raycast(marioState.pos, dir)
    #     # print(new_point, pos, dir)
    #     points_array.append(new_point)
    #     dir[2] += 10
    # points_array = np.array(points_array)
    # print(points_array)
    # pos_arr = [marioState.pos for i in range(3000)]
    # dir_arr = [np.random.uniform(-1000, 1000, size=3) for i in range(3000)]
    # points_array = game.get_raycasts(pos_arr, dir_arr)
    




    new_pts = game.get_raycast_sphere(0, amount=1000)
    points_array = np.concatenate((points_array, new_pts), axis=0)
    points_array = points_array[-10000:]

    if len(points_array) == 0:
        continue

    points_array = points_array[np.isfinite(points_array).all(axis=1)]
    if len(points_array) == 0:
        continue
    points_array = points_array[points_array[:, 0] != -8000]
    if len(points_array) == 0:
        continue

    # print(points_array)
    # reset scatter


    # Extract the x, y, and z coordinates
    x = points_array[:, 0]
    y = points_array[:, 2]
    z = points_array[:, 1]

    # Create a 3D scatter plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter(-x, y, z, s=1)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('3D Scatter Plot')

    # Set the axis limits to fit the data
    ax.set_xlim([-8000,8000])
    ax.set_ylim([-8000,8000])
    ax.set_zlim([-8000,8000])

    # Show the plot
    plt.draw()
    plt.pause(0.01)
    ax.cla()
    # plt.clf()

    
    print("--- %s seconds ---" % (time.time() - start_time))

    # print(networkPlayer.currCourseNum)
    # print([int(marioState.pos[i]) for i in range(3)])
