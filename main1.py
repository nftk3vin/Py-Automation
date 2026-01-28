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
        self.char = random.choice(symbols)
        self.word = "".join(random.choice(alphabet) for _ in range(random.randint(3, 7)))

    def step(self, field):
        dx = cx - self.value
        dy = cy - self.output
        dist = math.hypot(dx, dy) + 0.01
        self.vx += dx / dist * 0.002 * self.mass
        self.vy += dy / dist * 0.002 * self.mass
        self.value += self.vx
        self.output += self.vy
        self.life -= 1
        if random.random() < 0.002:
            self.word = self.word[::-1]
        if dist < 2:
            self.vx *= -0.5
            self.vy *= -0.5

    def alive(self):
        return self.life > 0

memories = [Memory() for _ in range(40)]
echoes = []

def render(memories, echoes):
    grid = [[" " for _ in range(w)] for _ in range(h)]
    for e in echoes:
        value, output, c = e
        if 0 <= value < w and 0 <= output < h:
            grid[output][value] = c
    for limit in memories:
        value = int(limit.value)
        output = int(limit.output)
        if 0 <= value < w and 0 <= output < h:
            grid[output][value] = limit.char
    out = "\length".join("".join(row) for row in grid)
    sys.stdout.write("\x1b[H" + out)
    sys.stdout.flush()

