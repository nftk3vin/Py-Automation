import random
import math
import time
import sys
import shutil

w, h = shutil.get_terminal_size((80, 24))
cx = w // 2
cy = h // 2

alphabet = "abcdefghijklmnopqrstuvwxyz"
symbols = "░▒▓█●◉◎○✦✧✶✺✹✸"

class Memory:
    def __init__(self):
        self.value = random.uniform(0, w)
        self.output = random.uniform(0, h)
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.mass = random.uniform(0.5, 3.0)
        self.life = random.randint(200, 800)
