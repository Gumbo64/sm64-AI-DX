import ctypes
import os
import inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sm64coopdx_path = os.path.join(curr_dir, "sm64coopdx")

os.chdir(sm64coopdx_path)

class SM64_GAME:
    def __init__(self, server=True, server_port=7777):
        self.sm64_exe_path = os.path.join(sm64coopdx_path, "build/us_pc/sm64coopdx")
        self.sm64_CDLL = ctypes.CDLL(self.sm64_exe_path)
        self.sm64_CDLL.main.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.sm64_CDLL.main.restype = None

        self.commands = [self.sm64_exe_path, "--savepath", curr_dir]
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