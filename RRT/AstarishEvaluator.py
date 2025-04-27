import numpy as np
import multiprocessing
# from AstarishWorker import AstarishWorker
from .PointPath import PointPath
from sm64env.sm64env_nothing import SM64_ENV_NOTHING
multiprocessing.set_start_method('spawn', force=True)
import time
import math
class AstarishEvaluator:
    def __init__(self):
        # self.num_workers = 8
        # self.starting_length = 100

        # self.seg_length = 10
        # self.starting_paths = 256
        # self.add_amount = 100

        # self.epsilon = 400
        # self.max_queue = 10000
        # self.discrete_mode = False

        # self.task_queue = multiprocessing.Queue()
        # self.result_queue = multiprocessing.Queue()
        # self.workers = []

        # for i in range(self.num_workers):
        #     worker = AstarishWorker(
        #         name=f"Worker{i+1}", 
        #         task_queue=self.task_queue, 
        #         result_queue=self.result_queue,
                
        #         multi_step=4,
        #         server_port=7777 + i,

        #     )
        #     self.workers.append(worker)
        #     worker.start()
        self.game = SM64_ENV_NOTHING(multi_step=4, server=True, server_port=7777)
        self.max_length = 200

    def evaluate(self, points) -> bool:
        pointPath = PointPath(points)

        done = False
        _, info = self.game.reset()
        position = info['pos']
        velocity = info['vel']
        globalAngle = info['angle']

        for i in range(self.max_length):
            goalPoint = pointPath.get_goalpoint()
            
            diff = goalPoint - position
            angleToGoal = np.arctan2(diff[2], diff[0])
            
            angle = angleToGoal + globalAngle
            # angle = 0
            print(angle / math.pi * 180)
            stickX = np.cos(angle) * 80
            stickY = np.sin(angle) * 80
            buttonA = 1 if goalPoint[1] > (position[1] + 10) else 0
            # buttonA = 0

            action = [(stickX, stickY), (buttonA, 0, 0)]
            print(action)
            obs, reward, done, truncated, info = self.game.step(action)
            time.sleep(0.0166 * 4)
            
            position = info['pos']
            velocity = info['vel']
            globalAngle = info['angle']
            reward, done = pointPath.update_goal(position)
            print(position)
            if done:
                break
        return done

        

