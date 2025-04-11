
from sm64env.sm64env_nothing import SM64_ENV_NOTHING

if __name__ == "__main__":
    # clear_sm64_exes()
    import numpy as np
    import torch
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt

    # def execute_path(path, frame_time=0):
    #     _, info = env.reset()
    #     print("STARTING")
    #     position = info['pos']
    #     for action in path:
    #         stick_actions = action[0:2] * 80
    #         button_actions = action[2:5] > 0.95
    #         action = (stick_actions, button_actions)

    #         obs, reward, done, truncated, info = env.step(action)
    #         position = info['pos']
    #         time.sleep(frame_time)

    from PATH_worker import SM64_WORKER

    multi_step = 1

    env = SM64_ENV_NOTHING(multi_step=multi_step, server=True, server_port=7777)


    # path = np.load('best_path_30014553.703223668.npy')
    # path = np.load('best_path/100_2488.2315575332086.npy')
    path = np.load('best_path_6749226.775843421.npy')
    # while True:
    #     execute_path(path, frame_time=0.0166 * multi_step)

    import multiprocessing
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    workers = []
    for i in range(8):
        worker = SM64_WORKER(
            name=f"Worker{i+1}", 
            multi_step=4,
            server_port=7777 + i, 
            task_queue=task_queue, 
            result_queue=result_queue
        )
        workers.append(worker)
        worker.start()

    # Generate and assign paths to workers
    for i in range(8):
        task_queue.put((path, 0))


    end_positions = []

    while len(end_positions) < 8:
        if not result_queue.empty():
            path, positions = result_queue.get()
            end_positions.append(positions)
            
            
    for path in end_positions:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 2], 'o-')
    plt.title("End Positions")
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    plt.show()




    # Stop the workers gracefully
    for worker in workers:
        task_queue.put(None)

    for worker in workers:
        worker.join()
