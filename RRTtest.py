from RRT.AstarishEvaluator import AstarishEvaluator
import numpy as np
evaluator = AstarishEvaluator()
while True:
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    result = evaluator.evaluate(points)
    print("Evaluation result:", result)