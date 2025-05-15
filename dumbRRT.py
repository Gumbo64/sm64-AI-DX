from RRT.AstarishEvaluator import AstarishEvaluator
from RRT.Util import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm

startPos = np.array([-6550.9297, 0, 6456.9297])
evaluator = AstarishEvaluator()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=90, azim=-90)

radius = 300
maximum = startPos
minimum = startPos

min_sample_dist = 100
max_sample_dist = 400

root = Node(None, startPos)
nodes = [root]
# root, nodes, minimum, maximum = load_root()

graph_nodes(ax, nodes)

with tqdm.tqdm() as pbar:
    while True:
        # Sample a random point in the environment
        random_point = sample_point(minimum, maximum, radius)

        # Find the closest node in the tree to the random point
        closest = closest_node(nodes, random_point)

        # Move towards the random point from the closest node
        new_position, d = point_towards(closest.get_position(), random_point)
        # if d > max_sample_dist or d < min_sample_dist:
        #     continue

        # If the evaluator agrees, create a new node and add it to the tree
        total_path = np.vstack((closest.p_path_to_me(), new_position))
        success, _ = evaluator.evaluate(total_path)
        if success:
            # Create a new node and add it to the tree
            new_node = Node(closest, new_position)
            nodes.append(new_node)

            minimum, maximum = update_min_max(new_node, minimum, maximum)

            # Update the progress bar   
            pbar.update(1)

            graph_nodes(ax, nodes)
            save_root(root, filename="root_node_RRT.npy")


