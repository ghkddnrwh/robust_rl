import enum
# import gym
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt

arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}
scale = 0.25
# scale = 1

ar =     [['R', 'D', 'L', 'L', 'L'],
          ['U', 'U', 'L', 'L', 'L'],
          ['U', 'U', 'L', 'L', 'L'],
          ['U', 'U', 'L', 'L', 'L'],
          ['U', 'U', 'L', 'L', 'L']]

fig, ax = plt.subplots(figsize = (6, 6))
# plt.subplot(6, 6, 1)
print("hello")
for r, row in enumerate(ar):
    for c, cell in enumerate(row):
        plt.arrow(c, 5-r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1)


plt.show()