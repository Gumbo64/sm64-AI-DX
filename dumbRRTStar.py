from RRT.AstarishEvaluator import AstarishEvaluator
from RRT.Util import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm

saveFilename = "root_node_RRTStar.npy"
loadFilename = None
# loadFilename = "root_node_walljumpandelevator.npy"
# loadFilename = "root_node_RRTStar.npy"

evaluator = AstarishEvaluator()
startPos = evaluator.start_pos()

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

border_radius = 300
maximum = startPos
minimum = startPos

step_size = 1000
min_improvement = 2

# min_sample_dist = 100
# max_sample_dist = 400

root = Node(None, startPos)
nodes = [root]
if loadFilename is not None:
    root, nodes, minimum, maximum = load_root(loadFilename)

graph_nodes_map(ax, nodes)

with tqdm.tqdm() as pbar:
    while True:
        # Sample a random point in the environment
        random_point = sample_point(minimum, maximum, border_radius)

        # Find the closest node in the tree to the random point
        closest = closest_node(nodes, random_point)

        # Move towards the random point from the closest node
        new_position, _ = point_towards(closest.get_position(), random_point, step_size=step_size)

        # If the evaluator agrees, create a new node and add it to the tree
        total_path = np.vstack((closest.p_path_to_me(), new_position))
        success, _ = evaluator.evaluate(total_path)
        if success:
            # Create a new node and add it to the tree
            new_node = Node(closest, new_position)
            minimum, maximum = update_min_max(new_node, minimum, maximum)

            # ######################## RRT Star specific
            # Get nodes within a certain radius, but not the new_node itself
            nearby_nodes = [node for node in nodes if np.linalg.norm(node.get_position() - new_node.get_position()) < step_size and node != closest]
            nodes.append(new_node)

            path_to_new_node = new_node.p_path_to_me()
            for node in nearby_nodes:
                if node.get_parent() is None:
                    # either the root node or a node that has been removed
                    continue
                
                distance_to_new_node = np.linalg.norm(node.get_position() - new_node.get_position())
                leaf_nodes = node.leaf_nodes()
                if new_node.cost() + distance_to_new_node + min_improvement * len(leaf_nodes) < node.cost():
                    # Check that all leaf nodes are unaffected by the reroute
                    for leaf_node in leaf_nodes:
                        # New path 
                        total_path = leaf_node.p_path_detour(new_node, node)

                        success, _ = evaluator.evaluate(total_path)
                        if not success:
                            # remove the node since you'd have to check all the nodes between the new node and the leaf node
                            node.remove_self(nodes)
                            break
                    else:
                        # new node is compatible! reroute to the new node
                        node.set_parent(new_node)
            # ########################
            # Update the progress bar
            pbar.update(1)

            # graph_nodes_map(ax, nodes)
            save_root(root, saveFilename)


