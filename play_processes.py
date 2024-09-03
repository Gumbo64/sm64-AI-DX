from sm64env.load_sm64_CDLL import SM64_GAME, clear_sm64_exes
from multiprocessing import Process
import random

clear_sm64_exes()

def play_game(id):
    game = SM64_GAME(server = (id == 0), server_port=7777)
    for _ in range(30000):
        stickX = random.randint(-80, 80)
        stickY = random.randint(-80, 80)
        buttonA, buttonB, buttonZ = random.choices([0, 1], weights=[0.9, 0.1], k=3)
        # buttonL = random.choices([0, 1], weights=[0.01, 0.99], k=1)[0]
        game.set_controller(stickX=stickX, stickY=stickY, buttonA=buttonA, buttonB=buttonB, buttonZ=buttonZ)
        game.step_game()

if __name__ == '__main__':
    num_processes = 3  # Specify the number of parallel processes you want to run
    processes = []

    for i in range(num_processes):
        p = Process(target=play_game, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
