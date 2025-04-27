import numpy as np

class PointPath:
    def __init__(self, points, radius=200):
        self.path = np.array(points)
        self.current_goal_idx = 0
        self.radius = radius

    def get_goalpoint(self) -> np.ndarray:
        return self.path[self.current_goal_idx]
    
    def update_goal(self, position) -> tuple[float, bool]:
        if self.current_goal_idx >= len(self.path):
            return 1000000.0, True
        goal = self.get_goalpoint()
        distance = np.linalg.norm(position - goal)
        if distance < self.radius:
            self.current_goal_idx += 1
            return 10, False

        reward = 1 - np.exp(-distance / self.radius)
        return reward, False



            

