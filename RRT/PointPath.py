import numpy as np

class PointPath:
    def __init__(self, points, radius=500):
        self.path = np.array(points)
        self.current_goal_idx = 0
        self.radius = radius
        self.goalTimes = []
        self.time = 0

    def get_goalpoint(self) -> np.ndarray:
        return self.path[self.current_goal_idx] if self.current_goal_idx < len(self.path) else self.path[-1]
    
    def get_times(self) -> np.ndarray:
        return np.array(self.goalTimes)

    # return reward, newGoal, done
    def update_goal(self, position) -> tuple[float, bool, bool]:
        if self.current_goal_idx >= len(self.path):
            return 1000000.0, False, True
        
        self.time += 1
        goal = self.get_goalpoint()
        distance = np.linalg.norm(position - goal)
        if distance < self.radius:
            self.current_goal_idx += 1
            
            if len(self.goalTimes) == 0:
                self.goalTimes.append(0 + self.time)
            else:
                self.goalTimes.append(self.goalTimes[-1] + self.time)
            self.time = 0 

            return 10, True, False

        reward = 1 - np.exp(-distance / self.radius)
        return reward, False, False



            

