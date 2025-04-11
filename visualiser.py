import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

import time
def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()        
        #plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


def visualise_game(marioStates, points_array, normals_array):
    if len(points_array) == 0:
        return
    
    for i, marioState in enumerate(marioStates):
        ax.scatter(-marioState.pos[0], marioState.pos[2], marioState.pos[1], c=('red' if i == 0 else 'blue'), marker='o', s=10, zorder=11)

    ax.scatter(-points_array[:, 0], points_array[:, 2], points_array[:, 1], s=5, c=(normals_array + 1) / 2, zorder=10)
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Mario 64 Point Cloud')

    # Set the axis limits to fit the data
    ax.set_xlim([-8192,8192])
    ax.set_ylim([-8192,8192])
    ax.set_zlim([-8192,8192])
    # Show the plot
    plt.draw()
    # plt.pause(0.001)
    ax.clear()



def visualise_game_tokens(tokens, pause_time=0.1):
    # if pause_time == 0:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # pos_scaler = 8192
    # vel_scaler = 50


    mario_token = tokens[0]
    
    ax.scatter(-mario_token[0], mario_token[2], mario_token[1], c='red', marker='o', s=10, zorder=11)


    point_tokens = tokens[1:]
    # point_tokens[:, 3:6] *= pos_scaler
    # print(point_tokens[:, 3:6])
    normalised_colours = point_tokens[:, 3:6] * 0.9999
    # print(normalised_colours)
    ax.scatter(-point_tokens[:, 0], point_tokens[:, 2], point_tokens[:, 1], c=(normalised_colours + 1)/2, marker='o', s=10, zorder=10)
    # ax.scatter(-point_tokens[:, 0], point_tokens[:, 2], point_tokens[:, 1], c="blue", marker='o', s=10, zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Mario 64 Point Cloud')

    # Set the axis limits to fit the data
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    # ax.set_xlim([-8192,8192])
    # ax.set_ylim([-8192,8192])
    # ax.set_zlim([-8192,8192])
    # Show the plot
    if pause_time == 0:
        plt.show()
    # else:
    ax.view_init(elev=90, azim=-90)
    plt.draw()
    # plt.pause(pause_time)
    # fig.canvas.draw_idle()
    # fig.canvas.start_event_loop(pause_time)
    plt.savefig('game.png')
    ax.clear()
    # plt.show()


def visualise_curiosity(curiosity, pause_time=1):
    # if pause_time == 0:
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Get the coordinates where indices is True
    F_indices = np.argwhere(curiosity.F > 0)
    # Plot circles at the True coordinates
    true_x, true_y, true_z = curiosity.multi_index_to_pos(F_indices).T

    visits = curiosity.F[F_indices[:, 0], F_indices[:, 1], F_indices[:, 2]]
    ax.scatter(-true_x, true_z, true_y, c=visits/curiosity.max_visits, cmap='coolwarm', marker='o', alpha=0.2, s=(min(curiosity.chunk_y_size, curiosity.chunk_xz_size)**2)/4, zorder=10)
    # Set the plot limits
    ax.set_xlim(-curiosity.bounding_size, curiosity.bounding_size)
    ax.set_ylim(-curiosity.bounding_size, curiosity.bounding_size)
    ax.set_zlim(-curiosity.bounding_size, curiosity.bounding_size)
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Mario 64 Curiosity')

    # plt.savefig('curiosity.png')

    with open('picklePlot.pickle', 'wb') as file:
        pickle.dump(fig, file)
    # Show the plot
    # if pause_time == 0:
    # plt.show(block=False)
    # else:
    # plt.draw()
    # plt.pause(pause_time)
    # ax.clear()
    # # 

def visualise_dfs_time(dfs, max_time):

    # Get the coordinates where indices is True
    F_indices = np.argwhere(dfs.F_time != np.inf)
    # Reduce the number of points by sampling
    sampled_indices = F_indices[np.random.choice(F_indices.shape[0], size=F_indices.shape[0] // 100, replace=False)]
    # Plot circles at the sampled coordinates
    true_x, true_y, true_z = dfs.multi_index_to_pos(sampled_indices).T

    visits = dfs.F_time[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]
    ax.scatter(-true_x, true_z, true_y, c=visits/max_time, cmap='coolwarm', marker='o', alpha=0.2, s=(min(dfs.chunk_y_size, dfs.chunk_xz_size)**2)/16, zorder=10)
    # Set the plot limits
    ax.set_xlim(-dfs.bounding_size, dfs.bounding_size)
    ax.set_ylim(-dfs.bounding_size, dfs.bounding_size)
    ax.set_zlim(-dfs.bounding_size, dfs.bounding_size)
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Mario 64 Curiosity')

    # plt.savefig('curiosity.png')
    plt.show()
    # with open('picklePlot.pickle', 'wb') as file:
    #     pickle.dump(fig, file)