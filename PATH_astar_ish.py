
from PATH_worker import SM64_WORKER, clamp_stick, generate_path, generate_discrete_path, action_book

import itertools

if __name__ == "__main__":
    import numpy as np
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    import torch
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)





    Q = {}

    def distance_to_goal(position):
        goalPos = np.array([5524.7285, 3072, 1187.4833])
        return np.sum(np.square(position - goalPos))

    num_workers = 8
    starting_paths = 256
    starting_length = 100

    seg_length = 10
    add_amount = 100

    max_queue = 10000

    discrete_mode = False

    epsilon = 400

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

    # Generate and assign paths to workers
    for i in range(starting_paths):
        if discrete_mode:
            start_path = generate_discrete_path(starting_length)
        else:
            start_path = generate_path(starting_length)
        task_queue.put((start_path, 0))

    prev_best = np.inf

    try:
        with tqdm() as iterbar:
            while True:
                if not result_queue.empty():
                    path, positions = result_queue.get()
                    
                    
                    d = distance_to_goal(positions[-1])

                    if d < prev_best + epsilon:
                        if d < prev_best:
                            while task_queue.qsize() > max_queue:
                                task_queue.get()

                            if prev_best != np.inf:
                                os.remove(f'best_path_{prev_best}.npy')
                                
                            prev_best = d
                            np.save(f'best_path_{prev_best}.npy', path)

                        for _ in range(add_amount):
                            if discrete_mode:
                                ext_path = generate_discrete_path(seg_length)
                            else:
                                ext_path = generate_path(seg_length)
                            
                            new_path = np.append(path, ext_path, axis=0)
                            
                            task_queue.put((new_path, 0))

                    iterbar.set_postfix(queue_size=task_queue.qsize()) 
                    iterbar.update(1)


    except KeyboardInterrupt:
        for worker in workers:
            worker.stop()
            worker.join()

    # Stop the workers gracefully
    for worker in workers:
        task_queue.put(None)

    for worker in workers:
        worker.join()


# with tqdm() as iterbar:
#     while True:
#         path = generate_path(length)
#         end_pos = execute_path(path)

#         if best_fitness is None or fitness > best_fitness:
#             if best_path is not None:
#                 os.remove(f'best_path{best_fitness}.pt')

#             best_path = path
#             best_fitness = fitness
#             torch.save(best_path, f'best_path{best_fitness}.pt')
#         iterbar.set_postfix(best_fitness=best_fitness)
#         iterbar.update(1)



