import pynput
import ctypes
from time import time
from time import sleep
import mss

sct = mss.mss()
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
SendInput = ctypes.windll.user32.SendInput

def set_pos(x, y):
    x = 1 + int(x * 65536./Wd)
    y = 1 + int(y * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

# tic = time()
sleep(1)
set_pos(641, 386)
# set_pos(2500, 200)
# print(f"took {(time() - tic)*1000} ms")