from RRT.AstarishEvaluator import AstarishEvaluator
import numpy as np
evaluator = AstarishEvaluator()

d = 1000

p1 = np.array([-6550.9297, 0, 6456.9297])
p2 = p1 + np.array([0, 0, -d])
p3 = p1 + np.array([10*d, 800, -d])
p4 = p1 + np.array([10*d, 800, 0])
p5 = p1

while True:
    points = np.array([p1, p2, p3, p4, p5])
    result = evaluator.evaluate(points)
    print("Evaluation result:", result)