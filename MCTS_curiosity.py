
from PATH_worker import SM64_WORKER, clamp_stick, generate_path, generate_discrete_path, action_book
from sm64env import curiosity_util
from sm64env.load_sm64_CDLL import clear_sm64_exes

import itertools
import matplotlib.pyplot as plt
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx

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

    

    class Node:
        def __init__(self, action, parent=None):
            self.parent = parent
            self.path_depth = 0
            if self.parent is not None:
                self.path_depth = self.parent.path_depth + 1

            self.children = []
            self.chosen_times = 0

            self.action = action

            self.curiosity = curiosity_util.CURIOSITY(max_visits=2048)

    
        def get_path(self):
            if self.parent is None:
                return [] # root node has no action
            return self.parent.get_path() + [self.action]     

        def is_leaf(self):
            return len(self.children) == 0
        
        
        def calculate_score(self):
            # use curiosity discretised visit space + entropy formula
            # consider also factoring in self.rollout_len, and len(self.children)

            # apparently you can normalise entropy by /ln(n) where n is the number of non-zero voxels
            # try with and without normalisation though idk which is better


            # n = self.chosen_times + num_nonzero_voxels
            
            if self.chosen_times == 0:
                return np.inf
            
            # return self.curiosity.entropy() / len(self.children)
            # return self.curiosity.entropy() * num_nonzero_voxels / n

            # return self.curiosity.entropy() + np.sqrt(2 * np.log(num_nonzero_voxels) / self.chosen_times)
            entropy, terms = self.curiosity.entropy()

            entropy_norm = entropy / np.log(terms)

            # return entropy_norm + np.sqrt(2 * np.log(terms) / self.chosen_times)
            # return - entropy / np.log(self.chosen_times)
            return entropy


    def traverse(node):
        if node.is_leaf():
            return node

        best_score = node.calculate_score()
        best_child = node
        for child in node.children:
            score = child.calculate_score()
            if score > best_score:
                best_score = score
                best_child = child
        
        if best_child == node:
            return node

        return traverse(best_child)
            

    def expansion(node):
        new_action = generate_path(1)[0]
        new_node = Node(new_action, parent=node)
        node.children.append(new_node)
        return new_node
    
    def simulation_and_backprop(node, num_rollouts, rollout_length):
        path = node.get_path()

        for _ in range(num_rollouts):
            task_queue.put((path, rollout_length))

        num_completed = 0
        while num_completed < num_rollouts:
            if not result_queue.empty():
                _, rollout_positions = result_queue.get()
                # node.rollout_positions = np.append(node.rollout_positions, rollout_positions, axis=0)
                # node.curiosity.add_circles(rollout_positions)
                backpropagation(node, rollout_positions)
                num_completed += 1

    def backpropagation(node, rollout_positions):
        node.curiosity.add_circles(rollout_positions)
        node.chosen_times += 1
        if node.parent is not None:
            backpropagation(node.parent, rollout_positions)

    def graph_tree(root):
        G = nx.DiGraph()
        nodes = [root]
        labels = {}
        depths = {}

        while nodes:
            node = nodes.pop()
            for child in node.children:
                G.add_edge(node, child)
                nodes.append(child)
                labels[child] = f"{child.calculate_score():.2f}"
                depths[child] = child.path_depth

        pos = graphviz_layout(G, prog="dot")
        plt.figure(figsize=(12, 8))

        # Get the depth of each node
        depth_values = [depths.get(node, 0) for node in G.nodes()]
        max_depth = max(depth_values) if depth_values else 1

        # Normalize depth values to range [0, 1] for colormap
        normalized_depths = [depth / max_depth for depth in depth_values]

        nx.draw(G, pos, with_labels=False, node_size=500, node_color=normalized_depths, cmap=plt.cm.cool, font_size=10, font_color="black")
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=plt.gca(), label='Depth from root')
        plt.savefig(f"tree.png")
        plt.close()

        


    num_workers = 8
    
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

    root = Node(None)
    simulation_and_backprop(root, 8, 64)

    try:
        for iter in tqdm(itertools.count()):
            best_node = traverse(root)
            new_node = expansion(best_node)
            simulation_and_backprop(new_node, 8, 64)

            if iter % 10 == 0:
                graph_tree(root)
                np.save(f'best_path/{iter}.npy', np.array(best_node.get_path()))

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



