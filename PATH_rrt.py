
from PATH_worker import SM64_WORKER, clamp_stick, generate_path, generate_discrete_path, action_book

import itertools
from scipy.spatial import KDTree

if __name__ == "__main__":
    import numpy as np
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    import torch
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)




    def distance_to_goal(position, goalPos):
        return np.sqrt(np.sum(np.square(position - goalPos)))

    num_workers = 8
    starting_paths = 256
    starting_length = 20

    seg_length = 1
    attempt_amount = 16

    epsilon = 0

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    workers = []
    for i in range(num_workers):
        worker = SM64_WORKER(
            name=f"Worker{i+1}", 
            multi_step=4,
            server_port=7777 + i, 
            task_queue=task_queue, 
            result_queue=result_queue
        )
        workers.append(worker)
        worker.start()

    
    def graph_V(V):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(V[:, 0], V[:, 1], V[:, 2])
        plt.savefig('graph.png')

    def asCloseAsPossible(path, goalPos, maxAttempts, segLength):
        for _ in range(maxAttempts):
            new_path = np.append(path, generate_discrete_path(segLength), axis=0)
            task_queue.put((new_path, 0))

        best_path = None
        best_distance = np.inf
        best_pos = None

        with tqdm(total=maxAttempts, leave=False) as iterbar:
            while True:
                if not result_queue.empty():
                    path, positions = result_queue.get()
                    
                    d = distance_to_goal(positions[-1], goalPos)

                    if d < best_distance:
                        best_distance = d
                        best_pos = positions[-1]
                        best_path = path
                        # best_positions = positions


                    if task_queue.qsize() == 0:
                        break
                    iterbar.update(1)

        return best_path, best_pos, best_distance

    class Node:
        def __init__(self, parent, path, pos):
            self.parent = parent
            self.path = path
            self.pos = pos
            self.children_ids = []
        
        def distance(self, pos):
            return np.linalg.norm(self.pos - pos)

    def nearest_node(V, pos):
        distances = np.linalg.norm(V - pos, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]

    


    start_pos = None
    start_path = generate_discrete_path(starting_length)

    task_queue.put((start_path, 0))
    while True:
        if not result_queue.empty():
            path, positions = result_queue.get()
            start_pos = positions[-1]
            break

    V = np.array([start_pos])
    N = {0: Node(None, start_path, start_pos)}

    try:
        for iter in tqdm(itertools.count()):
            rand_pos = np.array([
                np.random.uniform(-8000, 8000),
                # np.random.uniform(0, 800),
                0,
                np.random.uniform(-8000, 8000)
            ])
            x_nearest_idx, x_nearest_distance = nearest_node(V, rand_pos)
            x_nearest = N[x_nearest_idx]

            best_path, best_pos, best_distance = asCloseAsPossible(x_nearest.path, rand_pos, attempt_amount, seg_length)
            # needs to be epsilon closer than the nearest node to be added
            
            if best_distance < x_nearest_distance + epsilon:
                new_node = Node(x_nearest, best_path, best_pos)

                x_nearest.children_ids.append(len(N))
                N[len(N)] = new_node
                V = np.append(V, [rand_pos], axis=0)

                graph_V(V)
                
                
        

    except KeyboardInterrupt:
        for worker in workers:
            worker.stop()
            worker.join()

    # Stop the workers gracefully
    for worker in workers:
        task_queue.put(None)

    for worker in workers:
        worker.join()


