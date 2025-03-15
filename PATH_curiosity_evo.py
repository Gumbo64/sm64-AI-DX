
from PATH_worker import SM64_WORKER, clamp_stick, generate_path, generate_discrete_path, action_book
from sm64env import curiosity_util
from sm64env.load_sm64_CDLL import clear_sm64_exes

import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import random

if __name__ == "__main__":
    clear_sm64_exes()

    import numpy as np
    import time
    from tqdm import tqdm
    import os
    import matplotlib.pyplot as plt
    import torch
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    num_workers = 8

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

    game_length = 2000

    class Run:
        def __init__(self, path=None, score=None):
            if path is None:
                self.path = generate_discrete_path(game_length)
            else:
                self.path = path


            self.score = score

    population_size = 64

    population = [Run() for _ in range(population_size)]

    iter = 0
    try:
        with tqdm() as iterbar:
            while True:
                # Evaluate the population

                evaluated_population = []
                n_operations = 0
                for run in population:
                    if run.score is None:
                        task_queue.put((run.path, 0))
                        n_operations += 1
                    else:
                        evaluated_population.append(run)
                
                i = 0
                with tqdm(total=n_operations, leave=False) as pbar:
                    while i < n_operations:
                        if not result_queue.empty():
                            # Collect new data
                            path, score = result_queue.get()
                            evaluated_population.append(Run(path, score))
                            i += 1
                            pbar.update(1)


                population = evaluated_population

                # Sort the population
                population.sort(key=lambda run: run.score, reverse=True)

                iterbar.set_postfix(score=population[0].score)
                iterbar.update(1)
                iter += 1

                if iter % 10 == 0:
                    run = population[0]
                    np.save(f'best_path/{iter}_{run.score}.npy', np.array(run.path))


                # Kill the bottom half
                population = population[:population_size//2]

                # Breed the top half
                new_population = []
                for i in range(population_size//2):
                    parent1 = random.choice(population)
                    parent2 = random.choice(population)
                    child_path = np.array([p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1.path, parent2.path)])
                    new_population.append(Run(child_path))
                
                population += new_population

                # Mutate the population
                for run in population:
                    # 10% chance to mutate
                    if random.random() < 0.1:
                        g = random.sample(range(game_length), 2)
                        seg_start, seg_end = min(g), max(g)

                        new_path = run.path.copy()
                        new_path[seg_start:seg_end] = generate_path(seg_end - seg_start)
                        population[i] = Run(new_path)
                        



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



