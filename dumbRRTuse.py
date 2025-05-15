from RRT.AstarishEvaluator import AstarishEvaluator
from RRT.Util import *


# filename = "root_node_RRTStar copy 2.npy"
evaluator = AstarishEvaluator()

# startPos = np.array([-6550.9297, 0, 6456.9297])
startPos = evaluator.start_pos()

combinations = [
    {
        "filename": "root_node_throughgate.npy",
        "goals": [
            # np.array([-8000, 900, 0])
        ]
    },
    {
        "filename": "root_node_walljumpandelevator.npy",
        "goals": [
            np.array([-8000, 900, 0]),
            # np.array([8000, 900, 0]),
        ]
    },
    {
        "filename": "root_node_RRTStar copy.npy",
        "goals": [
            # np.array([8000, 900, 8000]),
            # np.array([8000, 900, -8000]),
        ]
    }
]


for c in combinations:
    root, nodes, minimum, maximum = load_root(c["filename"])

    for goal in c["goals"]:
        print(f"Evaluating {c['filename']} with goal {goal}")
        closest = closest_node(nodes, goal)
        total_path = closest.p_path_to_me()
        graph_path_map(total_path, nodes)
        print(evaluator.evaluate(total_path, delay=0.016666 * 4))

# radius = 300

# while True:
#     # Sample a random point in the environment
#     random_point = sample_point(minimum, maximum, radius)
#     # random_point = np.array([-8000, 900, 0])

#     # Find the closest node in the tree to the random point
#     closest = closest_node(nodes, random_point)

#     total_path = closest.p_path_to_me()
#     # graph_path(total_path, nodes)
#     print(evaluator.evaluate(total_path, delay=0.016666 * 4))



