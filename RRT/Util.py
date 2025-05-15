import numpy as np
import matplotlib.pyplot as plt
import os

class Node:
    def __init__(self, parent, position):
        self.position = position
        self.children = []
        self.parent = parent
        parent.add_child(self) if parent is not None else None
        self.last_cost = np.inf


    def remove_self(self, nodes) -> None:
        for child in self.children[:]: # shallow copy to avoid modifying the list while iterating
            child.remove_self(nodes)

        nodes.remove(self)
        self.parent.remove_child(self)
        self.parent = None
        assert len(self.children) == 0, "Node has children after removal"
        assert self not in nodes, "Node is still in the list after removal"
        assert self.parent is None, "Node still has a parent after removal"

    def get_parent(self) -> 'Node':
        return self.parent

    def set_parent(self, parent_node) -> None:
        if self.parent is None:
            raise ValueError("Node has no current parent")
        self.parent.remove_child(self)
        self.parent = parent_node
        self.parent.add_child(self)

    def add_child(self, child_node) -> None:
        self.children.append(child_node)
    def remove_child(self, child_node) -> None:
        self.children.remove(child_node)
    def get_children(self) -> list['Node']:
        return self.children
    def get_position(self) -> np.ndarray:
        return self.position

    def update_times(self, times) -> None:
        self.time = times[-1]
        if self.parent is None and times != []:
            raise ValueError("Node has no parent but has spare times")
        if self.parent is not None and times == []:
            raise ValueError("Node has a parent but no spare times")
        self.parent.update_times(times[:-1])

    def path_to_me(self) -> list['Node']:
        if self.parent is None:
            return [self]
        return self.parent.path_to_me() + [self]
    
    def path_detour(self, switch_node, ancestor) -> list['Node']:
        # switch_node is the new_node that ISN'T our ancestor
        # ancestor is our ancestor node that is after the switch_node

        # ie root -> ... -> switch_node -(DETOUR)> ancestor -> ... -> self

        if self == ancestor:
            return switch_node.path_to_me() + [self]
        return self.parent.path_detour(switch_node, ancestor) + [self]

    def p_path_to_me(self) -> np.ndarray:
        nodes = self.path_to_me()
        return nodes_positions(nodes)
    
    def p_path_detour(self, switch_node, ancestor) -> np.ndarray:
        nodes = self.path_detour(switch_node, ancestor)
        return nodes_positions(nodes)
    
    def d_path_to_me(self) -> float:
        nodes = self.path_to_me()
        return nodes_distance(nodes)
    
    def d_path_detour(self, switch_node, ancestor) -> float:
        nodes = self.path_detour(switch_node, ancestor)
        return nodes_distance(nodes)


    def leaf_nodes(self) -> list['Node']:
        if self.children == []:
            return [self]
        leafs = []
        for child in self.children:
            leafs.extend(child.leaf_nodes())
        return leafs

    def cost(self) -> float:
        if self.parent is None:
            self.last_cost = 0
        else:
            self.last_cost = np.linalg.norm(self.position - self.parent.get_position()) + self.parent.cost()
        return self.last_cost
    
    def get_last_cost(self) -> float:
        if not hasattr(self, 'last_cost'):
            return self.cost()
        return self.last_cost



def sample_point(minimum, maximum, r) -> np.ndarray:
    # Sample a random point in the environment

    lower_bound = minimum - np.array([r, r, r])
    lower_bound = np.maximum(lower_bound, np.array([-8000, -8000, -8000]))

    upper_bound = maximum + np.array([r, r, r])
    upper_bound = np.minimum(upper_bound, np.array([8000, 8000, 8000]))

    return np.random.uniform(lower_bound, upper_bound)
    # return np.random.uniform(low=-8000, high=8000, size=3)

def point_towards(start: np.ndarray, goal: np.ndarray, step_size: float = 300) -> tuple[np.ndarray, float]:
    direction = goal - start
    distance = np.linalg.norm(direction)
    if distance == 0:
        return start, distance
    return start + (direction / distance) * min(distance, step_size), distance


def closest_node(nodes, point) -> Node:
    return min(nodes, key=lambda node: np.linalg.norm(node.get_position() - point))


def save_root(root, filename="root_node.npy") -> None:
    # Save the nodes to a file
    os.makedirs("RRTRoots", exist_ok=True)
    with open("RRTRoots/" + filename, "wb") as f:
        np.save(f, root)

def load_root(filename="root_node.npy"):
    root = None

    if os.path.exists("RRTRoots/" + filename):
        with open("RRTRoots/" + filename, "rb") as f:
            root = np.load(f, allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"File {filename} not found in RRTRoots directory.")

    #add the root and all its children to the nodes list
    nodes = [root]
    def add_children(node):
        for child in node.get_children():
            nodes.append(child)
            add_children(child)
    add_children(root)

    minimum = np.min([node.get_position() for node in nodes], axis=0)
    maximum = np.max([node.get_position() for node in nodes], axis=0)

    return root, nodes, minimum, maximum

def update_min_max(new_node, minimum, maximum) -> tuple[np.ndarray, np.ndarray]:
    # Update the minimum and maximum bounds based on the new node's position
    minimum = np.minimum(minimum, new_node.get_position())
    maximum = np.maximum(maximum, new_node.get_position())
    return minimum, maximum

def nodes_positions(nodes) -> np.ndarray:
    # Get the positions of all nodes
    positions = np.array([node.get_position() for node in nodes])
    return positions

def nodes_distance(nodes) -> float:
    distance = 0
    for i in range(len(nodes) - 1):
        n1 = nodes[i]
        n2 = nodes[i + 1]
        distance += np.linalg.norm(n2.get_position() - n1.get_position())
    return distance

def graph_nodes(ax, nodes) -> None:
    # Create a graph of nodes

    ax.clear()
    
    ax.set_xlim([-8000, 8000])
    ax.set_ylim([-8000, 8000])
    ax.set_zlim([-8000, 8000])
    ax.view_init(elev=90, azim=-90)

    max_depth = 64000
    for node in nodes:
        if node.parent is not None:
            color = plt.cm.hsv(np.log2(1 + (np.clip(node.parent.position[1] + 8000, 1e-6, 8000))/16000))  # Normalize position[1] to [0, 1] range
            # color = plt.cm.hsv(np.log2(1 + (np.clip(node.cost(), 0, max_depth))/max_depth))
            # color = plt.cm.hsv(np.clip(node.cost(), 0, max_depth)/max_depth)

            ax.plot([-node.position[0], -node.parent.position[0]],
                    [node.position[2], node.parent.position[2]],
                    [node.position[1], node.parent.position[1]],
                    color=color)
    plt.draw()
    plt.pause(0.01)

def graph_nodes_map(ax, nodes) -> None:
    ax.clear()
    ax.set_xlim([-8000, 8000])
    ax.set_ylim([-8000, 8000])
    # Load the map image as the background
    map_image = plt.imread("map_BOB.png")
    ax.imshow(map_image, extent=[-8000, 8000, -8000, 8000], origin='upper')
    max_depth = 64000
    for node in nodes:
        if node.parent is not None:
            # color = plt.cm.hsv(np.log2(1 + (np.clip(node.position[1] + 8000, 1e-6, 8000))/16000))
            # color = plt.cm.hsv(np.log2(1 + (np.clip(node.parent.get_last_cost(), 0, max_depth))/max_depth))
            color = plt.cm.hsv(np.clip(node.parent.get_last_cost(), 0, max_depth)/max_depth)
            ax.plot([-node.position[0], -node.parent.position[0]],
                    [node.position[2], node.parent.position[2]],
                    # [node.position[1], node.parent.position[1]],
                    color=color)
    plt.draw()
    plt.pause(0.01)



def graph_path(path, nodes) -> None:
    # Create a graph of nodes

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-8000, 8000])
    ax.set_ylim([-8000, 8000])
    ax.set_zlim([-8000, 8000])

    # Set the initial point of view from above
    ax.view_init(elev=90, azim=-90)

    for node in nodes:
        if node.parent is not None:
            color = plt.cm.hsv(np.log2(1 + (np.clip(node.parent.get_last_cost() + 8000, 1e-6, 8000))/16000))  # Normalize position[1] to [0, 1] range
            ax.plot([-node.position[0], -node.parent.position[0]],
                    [node.position[2], node.parent.position[2]],
                    [node.position[1], node.parent.position[1]],
                    color=color)

    for i in range(len(path) - 1):
        n1 = path[i]
        n2 = path[i + 1]

        color = plt.cm.viridis((n2[1] + 8000) / 16000)  # Normalize position[1] to [0, 1] range
        ax.plot([-n2[0], -n1[0]],
                [n2[2], n1[2]],
                [n2[1], n1[1]],
                color=color)

    # Plot the input path
    if path is not None and len(path) > 1:
        ax.plot(-path[:, 0], path[:, 2], path[:, 1], color='red', linewidth=2)
    plt.draw()
    # plt.pause(0.01)
    plt.show()



def graph_path_map(path, nodes) -> None:
    # Create a graph of nodes

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-8000, 8000])
    ax.set_ylim([-8000, 8000])

    # Load the map image as the background
    map_image = plt.imread("map_BOB.png")
    ax.imshow(map_image, extent=[-8000, 8000, -8000, 8000], origin='upper')

    for node in nodes:
        if node.parent is not None:
            color = plt.cm.hsv(np.log2(1 + (np.clip(node.parent.get_last_cost() + 8000, 1e-6, 8000))/16000))  # Normalize position[1] to [0, 1] range
            ax.plot([-node.position[0], -node.parent.position[0]],
                    [node.position[2], node.parent.position[2]],
                    # [node.position[1], node.parent.position[1]],
                    color=color)

    for i in range(len(path) - 1):
        n1 = path[i]
        n2 = path[i + 1]

        color = plt.cm.viridis((n2[1] + 8000) / 16000)  # Normalize position[1] to [0, 1] range
        ax.plot([-n2[0], -n1[0]],
                [n2[2], n1[2]],
                # [n2[1], n1[1]],
                color=color)

    # Plot the input path
    if path is not None and len(path) > 1:
        ax.plot(-path[:, 0], path[:, 2], color='red', linewidth=2)
    plt.draw()
    # plt.pause(0.01)
    plt.show()