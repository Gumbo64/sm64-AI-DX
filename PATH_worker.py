import multiprocessing
import time
from sm64env.sm64env_nothing import SM64_ENV_NOTHING
from sm64env.curiosity_util import CURIOSITY

multiprocessing.set_start_method('spawn', force=True)
import numpy as np

action_book = []
action_book_weights = []
for buttonA in [0, 1]:
    for buttonB in [0, 1]:
        for buttonZ in [0, 1]:
            for angle in [0, 45, -45, 90, -90, 135, -135, 180]:
                prob = 0.125
                prob *= 0.05 if buttonA == 1 else 0.95
                prob *= 0.05 if buttonB == 1 else 0.95
                prob *= 0.05 if buttonZ == 1 else 0.95

                stickX = np.sin(np.deg2rad(angle))
                stickY = np.cos(np.deg2rad(angle))
                action_book.append(np.array([stickX,stickY,buttonA, buttonB, buttonZ]))
                action_book_weights.append(prob)

action_book = np.array(action_book)
action_book_weights = np.array(action_book_weights)

def clamp_stick(array):
    # Clamp the stick length, but maintain the direction
    vec_length = np.linalg.norm(array, axis=1, keepdims=True)
    epsilon = 1e-8
    norm_vec = array / (vec_length + epsilon)
    clamped_length = np.abs(np.clip(vec_length, -80, 80))

    return norm_vec * clamped_length

def generate_path(length):
    if length == 0:
        return np.empty((0, 5))
    path = np.random.rand(length, 5) * 2 - 1
    path[:, 2] += 0.5
    path[:, 0:2] = clamp_stick(path[:, 0:2])
    return path

def generate_discrete_path(length):
    if length == 0:
        return np.empty((0, 5))
    # path_actions = np.random.randint(0, len(action_book), length)
    path_actions = np.random.choice(len(action_book), length, p=action_book_weights)
    path = action_book[path_actions]
    path[:, 0:2] = clamp_stick(path[:, 0:2])
    return path

class SM64_WORKER(multiprocessing.Process):
    def __init__(self, name, task_queue, result_queue, multi_step=1, server_port=7777):
        multiprocessing.Process.__init__(self)
        self.name = name
        self._stop_event = multiprocessing.Event()
        self.multi_step = multi_step
        self.env = SM64_ENV_NOTHING(server=True, server_port=server_port, multi_step=multi_step)
        self.task_queue = task_queue
        self.result_queue = result_queue
        
    def run(self):
        while not self._stop_event.is_set():
            if not self.task_queue.empty():
                task = self.task_queue.get()
                if task is None:
                    self.stop()
                    break

                path, rollout_length = task
                result = self.execute_path(path, rollout_length)
                self.result_queue.put(result)
            else:
                time.sleep(0.3)
                pass
        
    def execute_path(self, path, rollout_length):
        score = 0

        # c = CURIOSITY(max_visits=2048)
        # rollout_path = generate_path(rollout_length)
        rollout_path = generate_discrete_path(rollout_length)
        if len(path)==0:
            total_path = rollout_path
        else:
            total_path = np.concatenate((np.array(path), rollout_path), axis=0)
        
        positions = np.zeros((len(total_path), 3))

        _, info = self.env.reset()
        for i, action in enumerate(total_path):
            stick_actions = action[0:2] * 80
            button_actions = action[2:5] > 0.95
            action = (stick_actions, button_actions)
            _, _, _, _, info = self.env.step(action)

            position = info['pos']
            velocity = info['vel']
            # score += 1 - c.get_visits(position)/c.max_visits + np.linalg.norm(velocity) / 100
            # c.add_circle(position)
            positions[i] = position

        return path, positions
        # return path, score

        
    def stop(self):
        self._stop_event.set()
        

if __name__ == "__main__":
    from tqdm import tqdm

    


    multiprocessing.set_start_method('spawn', force=True)
    num_workers = 8
    n_tasks = 256
    length = 800
    rollout_length = 100

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    workers = []
    
    for i in range(num_workers):
        worker = SM64_WORKER(
            name=f"Worker{i+1}", 
            server_port=7777 + i, 
            task_queue=task_queue, 
            result_queue=result_queue
        )
        workers.append(worker)
        worker.start()
    
    start_time = time.time()
    
    try:
        # Generate and assign paths to workers
        for i in range(n_tasks):
            example_path = generate_path(length)
            task = (example_path, rollout_length)
            task_queue.put(task)
        

        
        results = []
        i = 0
        with tqdm(total=n_tasks) as pbar:
            while not task_queue.qsize() == 0:
                if not result_queue.empty():
                    path, rollout_positions = result_queue.get()
                    pbar.update(1)
                    # print(f"{len(results)} Path execution result: {result}")
                # time.sleep(0.1)
    
    except KeyboardInterrupt:
        for worker in workers:
            worker.stop()
            worker.join()
    
    # Stop the workers gracefully
    for worker in workers:
        task_queue.put(None)
    
    for worker in workers:
        worker.join()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")