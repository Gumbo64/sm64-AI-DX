import numpy as np
import matplotlib.pyplot as plt
import pickle

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)


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
    plt.pause(0.001)
    ax.clear()

def visualise_game_tokens(tokens, pause_time=0.1):
    # if pause_time == 0:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    pos_scaler = 8192
    vel_scaler = 50


    main_player_indice = np.where(tokens[:, 0] == 1)
    player_token_indices = np.where(tokens[:, 1] == 1)
    point_token_indices = np.where(tokens[:, 2] == 1)

    main_player_tokens = tokens[main_player_indice]
    main_player_tokens[:, 3:6] *= pos_scaler
    main_player_tokens[:, 6:9] *= vel_scaler
    ax.scatter(-main_player_tokens[:, 3], main_player_tokens[:, 5], main_player_tokens[:, 4], c='red', marker='o', s=10, zorder=11)

    player_tokens = tokens[player_token_indices]
    player_tokens[:, 3:6] *= pos_scaler
    player_tokens[:, 6:9] *= vel_scaler
    ax.scatter(-player_tokens[:, 3], player_tokens[:, 5], player_tokens[:, 4], c='blue', marker='o', s=10, zorder = 11)
    
    point_tokens = tokens[point_token_indices]
    point_tokens[:, 3:6] *= pos_scaler
    ax.scatter(-point_tokens[:, 3], point_tokens[:, 5], point_tokens[:, 4], c=(point_tokens[:, 6:9] + 1)/2, marker='o', s=10, zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Mario 64 Point Cloud')

    # Set the axis limits to fit the data
    ax.set_xlim([-8192,8192])
    ax.set_ylim([-8192,8192])
    ax.set_zlim([-8192,8192])
    # Show the plot
    if pause_time == 0:
        plt.show()
    # else:
    plt.draw()
    plt.pause(pause_time)
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

    plt.savefig('curiosity.png')
    with open('picklePlot.pickle', 'wb') as file:
        pickle.dump(fig, file)
    # Show the plot
    # if pause_time == 0:
    # plt.show(block=False)
    # else:
    # plt.draw()
    # plt.pause(pause_time)
    ax.clear()
    # 
