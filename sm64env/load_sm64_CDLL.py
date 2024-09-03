import ctypes
import os
import inspect
import shutil
from .sm64_structs import MarioState, NetworkPlayer
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sm64coopdx_path = os.path.join(curr_dir, "sm64coopdx")
build_dir = os.path.join(sm64coopdx_path, "build", "us_pc")

os.chdir(sm64coopdx_path)

def clear_sm64_exes():
    for file in os.listdir(build_dir):
        if file.startswith("sm64coopdx_"):
            os.remove(os.path.join(build_dir, file))

class SM64_GAME:
    def __init__(self, server=True, server_port=7777):
        base_sm64_exe_path = os.path.join(build_dir, "sm64coopdx")
        self.sm64_exe_path = os.path.join(build_dir, f"sm64coopdx_{str(id(self))}")
        shutil.copyfile(base_sm64_exe_path, self.sm64_exe_path)


        self.sm64_CDLL = ctypes.CDLL(self.sm64_exe_path, mode=ctypes.RTLD_LOCAL)

        self.sm64_CDLL.main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.sm64_CDLL.main.restype = None

        self.sm64_CDLL.step_game.argtypes = []
        self.sm64_CDLL.step_game.restype = None
        
        self.sm64_CDLL.set_controller.argtypes = [
            ctypes.c_int for _ in range(19)
        ]
        self.sm64_CDLL.set_controller.restype = None
        
        self.sm64_CDLL.get_mario_state.argtypes = [ctypes.c_int]
        self.sm64_CDLL.get_mario_state.restype = ctypes.POINTER(MarioState)

        self.sm64_CDLL.get_network_player.argtypes = [ctypes.c_int]
        self.sm64_CDLL.get_network_player.restype = ctypes.POINTER(NetworkPlayer)

        self.sm64_CDLL.raycast.argtypes = [ctypes.c_float * 3, ctypes.c_float * 3, ctypes.c_float * 3]
        self.sm64_CDLL.raycast.restype = None

        self.sm64_CDLL.raycasts.argtypes = [ctypes.POINTER(ctypes.c_float * 3), ctypes.POINTER(ctypes.c_float * 3), ctypes.POINTER(ctypes.c_float * 3), ctypes.c_int]
        self.sm64_CDLL.raycasts.restype = None

        self.sm64_CDLL.raycast_sphere.argtypes = [ctypes.POINTER(ctypes.c_float * 3), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.sm64_CDLL.raycast_sphere.restype = None

        self.commands = [self.sm64_exe_path, "--hide-loading-screen", "--skip-update-check", "--savepath", curr_dir]
        if server:
            self.commands += ["--server", str(server_port)]
        else:
            self.commands += ["--client", "localhost", str(server_port)]

        self.ctypes_commands = (ctypes.c_char_p * len(self.commands))()
        self.ctypes_commands[:] = [commands.encode('utf-8') for commands in self.commands]
        self.sm64_CDLL.main(len(self.commands), self.ctypes_commands)

    def step_game(self):
        # return self.sm64_CDLL.step_game()
        return self.sm64_CDLL.produce_one_frame()

    def set_controller(self,
        playerIndex = 0, stickX = 0, stickY = 0,
        buttonA = 0, buttonB = 0, buttonX = 0, buttonY = 0,
        buttonL = 0, buttonR = 0, buttonZ = 0, buttonStart = 0,
        buttonDU = 0, buttonDL = 0, buttonDR = 0, buttonDD = 0,
        buttonCU = 0, buttonCL = 0, buttonCR = 0, buttonCD = 0
    ):
        stickX = int(stickX)
        stickY = int(stickY)
        assert -80 <= stickX <= 80
        assert -80 <= stickY <= 80


        buttons = [buttonA, buttonB, buttonX, buttonY, buttonL, buttonR, buttonZ, buttonStart,
                   buttonDU, buttonDL, buttonDR, buttonDD, buttonCU, buttonCL, buttonCR, buttonCD]
        assert all(0 <= button <= 1 for button in buttons)
    
        self.sm64_CDLL.set_controller(
            playerIndex, stickX, stickY,
            buttonA, buttonB, buttonX, buttonY,
            buttonL, buttonR, buttonZ, buttonStart,
            buttonDU, buttonDL, buttonDR, buttonDD,
            buttonCU, buttonCL, buttonCR, buttonCD            
        )

    def get_mario_state(self, playerIndex):
        player_mario_state = self.sm64_CDLL.get_mario_state(playerIndex)
        player_mario_state = player_mario_state.contents
        return player_mario_state
    
    def get_network_player(self, playerIndex):
        network_player = self.sm64_CDLL.get_network_player(playerIndex)
        network_player = network_player.contents
        return network_player

    def get_raycast(self, pos, dir):
        ctypes_pos = (ctypes.c_float * 3)(*pos)
        ctypes_dir = (ctypes.c_float * 3)(*dir)
        ctypes_hitpos = (ctypes.c_float * 3)()
        self.sm64_CDLL.raycast(ctypes_hitpos, ctypes_pos, ctypes_dir)
        return np.array([ctypes_hitpos[0], ctypes_hitpos[1], ctypes_hitpos[2]])

    def get_raycasts(self, pos_arr, dir_arr):
        ctypes_pos_arr = (ctypes.c_float * 3 * len(pos_arr))()
        ctypes_dir_arr = (ctypes.c_float * 3 * len(dir_arr))()
        ctypes_hitpos_arr = (ctypes.c_float * 3 * len(pos_arr))()
        
        for i in range(len(pos_arr)):
            ctypes_pos_arr[i][:] = pos_arr[i]
            ctypes_dir_arr[i][:] = dir_arr[i]
        self.sm64_CDLL.raycasts(ctypes_hitpos_arr, ctypes_pos_arr, ctypes_dir_arr, len(pos_arr))
        return np.array([[ctypes_hitpos_arr[i][0], ctypes_hitpos_arr[i][1], ctypes_hitpos_arr[i][2]] for i in range(len(pos_arr))])
    
    def get_raycast_sphere(self, playerIndex, amount=3000, vecLength=10000):
        ctypes_hitpos_arr = (ctypes.c_float * 3 * amount)()
        self.sm64_CDLL.raycast_sphere(ctypes_hitpos_arr, amount, playerIndex, vecLength)
        return np.array([[ctypes_hitpos_arr[i][0], ctypes_hitpos_arr[i][1], ctypes_hitpos_arr[i][2]] for i in range(amount)])