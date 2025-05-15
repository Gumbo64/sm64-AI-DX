import numpy as np
import multiprocessing
# from AstarishWorker import AstarishWorker
import time
from .PointPath import PointPath
from sm64env.sm64env_nothing import SM64_ENV_NOTHING
multiprocessing.set_start_method('spawn', force=True)

class AstarishEvaluator:
    def __init__(self, goal_radius=300):
        self.game = SM64_ENV_NOTHING(multi_step=4, server=True, server_port=7777)
        self.goal_radius = goal_radius
        self.goal_timeout = 40

    def start_pos(self):
        _, info = self.game.reset()
        self.reset_policy()

        position = info['pos']
        return position
    # Decides what action to take deterministically
    def policy(self, info, goalPoint):
        position = info['pos']
        globalAngle = info['angle']

        diff = goalPoint - position
        angleToGoal = np.arctan2(diff[2], diff[0])
        
        angle = angleToGoal - globalAngle
        stickX = np.sin(angle) * 80
        stickY = np.cos(angle) * 80
        buttonA = not self.pressed_last
        if goalPoint[1] > (position[1]):
            buttonA = 0 if self.pressed_last else 1
            self.pressed_last = not self.pressed_last

        return [(stickX, stickY), (buttonA, 0, 0)]

    def reset_policy(self):
        self.pressed_last = False

    def evaluate(self, points, delay=0) -> tuple[bool, np.ndarray]:
        pointPath = PointPath(points)

        done = False
        _, info = self.game.reset()
        
        self.reset_policy()

        position = info['pos']
        lastGoalTime = 0

        while True:
            goalPoint = pointPath.get_goalpoint()
            
            action = self.policy(info, goalPoint)
            _, _, _, _, info = self.game.step(action)
            time.sleep(delay)
            
            position = info['pos']
            _, newGoal, done = pointPath.update_goal(position)
            if newGoal:
                lastGoalTime = 0
            else:
                lastGoalTime += 1
                if lastGoalTime > self.goal_timeout:
                    return False, pointPath.get_times()

            if done:
                return True, pointPath.get_times()

        

