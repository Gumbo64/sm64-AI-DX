import ctypes
import os
import inspect
import shutil
from .sm64_structs import MarioState, NetworkPlayer, Pixels
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sm64coopdx_path = os.path.join(curr_dir, "sm64coopdx")
build_dir = os.path.join(sm64coopdx_path, "build", "us_pc")

# os.chdir(sm64coopdx_path)

Vec3f = ctypes.c_float * 3

def clear_sm64_exes():
    for file in os.listdir(build_dir):
        if file.startswith("sm64coopdx_"):
            os.remove(os.path.join(build_dir, file))

class SM64_GAME:
    def __init__(self, server=True, server_port=7777, config_file="sm64config.txt"):
        base_sm64_exe_path = os.path.join(build_dir, "sm64coopdx")
        self.sm64_exe_path = os.path.join(build_dir, f"sm64coopdx_{str(id(self))}")
        shutil.copyfile(base_sm64_exe_path, self.sm64_exe_path)
        self.i = 10000

        self.sm64_CDLL = ctypes.CDLL(self.sm64_exe_path, mode=ctypes.RTLD_LOCAL)

        self.sm64_CDLL.main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.sm64_CDLL.main.restype = None

        self.sm64_CDLL.step_game.argtypes = [ctypes.c_int]
        self.sm64_CDLL.step_game.restype = None
        
        self.sm64_CDLL.set_controller.argtypes = [
            ctypes.c_int for _ in range(19)
        ]
        self.sm64_CDLL.set_controller.restype = None
        
        self.sm64_CDLL.get_mario_state.argtypes = [ctypes.c_int]
        self.sm64_CDLL.get_mario_state.restype = ctypes.POINTER(MarioState)

        self.sm64_CDLL.get_network_player.argtypes = [ctypes.c_int]
        self.sm64_CDLL.get_network_player.restype = ctypes.POINTER(NetworkPlayer)

        self.sm64_CDLL.raycast_with_normal.argtypes = [Vec3f, Vec3f, Vec3f, Vec3f]
        self.sm64_CDLL.raycast_with_normal.restype = None

        self.sm64_CDLL.raycast_sphere_with_normal.argtypes = [ctypes.POINTER(Vec3f), ctypes.POINTER(Vec3f), ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.sm64_CDLL.raycast_sphere_with_normal.restype = None

        self.sm64_CDLL.get_pixels_size.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.sm64_CDLL.get_pixels_size.restype = None

        # self.sm64_CDLL.get_pixels.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
        # self.sm64_CDLL.get_pixels.restype = None
        
        self.sm64_CDLL.get_pixels.argtypes = []
        self.sm64_CDLL.get_pixels.restype = ctypes.POINTER(Pixels)

        self.sm64_CDLL.get_lakitu_pos.argtypes = []
        self.sm64_CDLL.get_lakitu_pos.restype = ctypes.POINTER(Vec3f)

        self.sm64_CDLL.get_lakitu_yaw.argtypes = []
        self.sm64_CDLL.get_lakitu_yaw.restype = ctypes.c_int16


        self.commands = [self.sm64_exe_path, "--hide-loading-screen", "--skip-update-check",
                         "--savepath", curr_dir, 
                         "--configfile", config_file]
        if server:
            self.commands += ["--server", str(server_port)]
        else:
            self.commands += ["--client", "localhost", str(server_port)]

        self.ctypes_commands = (ctypes.c_char_p * len(self.commands))()
        self.ctypes_commands[:] = [commands.encode('utf-8') for commands in self.commands]
        self.sm64_CDLL.main(len(self.commands), self.ctypes_commands)

    def __del__(self):
        # Unload the shared library, deinitialise the game, delete the temporary executable
        # dlclose_func = ctypes.cdll.LoadLibrary('').dlclose
        # dlclose_func.argtypes = [ctypes.c_void_p]
        # handle = self.sm64_CDLL._handle

        self.sm64_CDLL.game_deinit()
        del self.sm64_CDLL
        # dlclose_func(handle)
        
        # Sadly this doesn't always work with async etc
        try:
            os.remove(self.sm64_exe_path)
        except OSError as e:
            # print(f"Error removing {self.sm64_exe_path}")
            pass

    def step_game(self, num_steps=1):
        return self.sm64_CDLL.step_game(num_steps)

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

    def get_raycast_with_normal(self, pos, dir):
        ctypes_pos = (Vec3f)(*pos)
        ctypes_dir = (Vec3f)(*dir)
        ctypes_hitpos = (Vec3f)()
        ctypes_normal = (Vec3f)()
        self.sm64_CDLL.raycast_with_normal(ctypes_hitpos, ctypes_normal, ctypes_pos, ctypes_dir)
        return np.array(ctypes_hitpos), np.array(ctypes_normal)

    def get_raycast_sphere_with_normal(self, amount=3000, maxRayLength=28000, cameraDirBiasFactor=0):
        ctypes_hitpos_arr = (Vec3f * amount)()
        ctypes_normal_arr = (Vec3f * amount)()
        self.sm64_CDLL.raycast_sphere_with_normal(ctypes_hitpos_arr, ctypes_normal_arr, amount, maxRayLength, cameraDirBiasFactor)
        return np.array(ctypes_hitpos_arr), np.array(ctypes_normal_arr)

    def get_pixels(self):
        # w = ctypes.c_int()
        # h = ctypes.c_int()
        # self.sm64_CDLL.get_pixels_size(ctypes.byref(w), ctypes.byref(h))
        # s = (w.value ) * (h.value) * 3
        # print(w, h, s, self.i)

        # pixels = (ctypes.c_ubyte * s)()
        ctypes_pixels = self.sm64_CDLL.get_pixels().contents

        w = ctypes_pixels.pixelsWidth
        h = ctypes_pixels.pixelsHeight

        # claude stopped weird padding misalignment
        row_bytes = w * 3
        padding = (4 - (row_bytes % 4)) % 4
        stride = row_bytes + padding
        img = np.ctypeslib.as_array(ctypes_pixels.pixels, shape=(h * stride,))
        idx = np.arange(h * row_bytes) + (np.arange(h) * padding).repeat(row_bytes)

        img = img[idx].reshape((h, w, 3))[:, :-padding, :]

        # flip image (cosmetic for human viewing, AI doesnt care)
        img = np.flipud(img).astype(np.uint8)
        
        return img

    def get_lakitu_pos(self):
        return np.array(self.sm64_CDLL.get_lakitu_pos().contents)
    
    def get_lakitu_yaw(self):
        return self.sm64_CDLL.get_lakitu_yaw()