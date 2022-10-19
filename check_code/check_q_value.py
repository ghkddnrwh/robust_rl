# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

r = 6
slippery = 0
scale = 0.25

action_to_str = ["U", "R", "D", "L"]
arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}

save_path_name = os.path.join("data", "iisl2", "test15")
env_name = 'CliffWalking-v0'


slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


simulation_name = "Robust_RL_R=" + str(r_list[r])
path_env_name = "FrozenLake-v1_slipery=" + str(slippery_list[slippery])

env_path = os.path.join(save_path_name, path_env_name, simulation_name, "q_value.npy")



q_value = np.load(env_path)
print(q_value.shape)

frozen_x_list = [8 * 2 + 3, 8*3+5, 8*4+3, 8*5+1, 8*5+2, 8*5+6, 8*6+4, 8*6+6, 8*7+3, 8*6+1]

q_value = q_value[1, :, :]

max_value = np.argmax(q_value, axis = 1)
str_list = []

for i in range(len(max_value)):
    str_list.append(action_to_str[max_value[i]])

str_list = np.reshape(str_list, (8, 8))

fig, ax = plt.subplots(figsize = (8, 8))
# plt.subplot(6, 6, 1)
# plt.plot(figsize = (4, 12))
print("hello")
for r, row in enumerate(str_list):
    for c, cell in enumerate(row):
        x_val = 8 * r + c
        if x_val in frozen_x_list:
            continue
        plt.arrow(c + 0.5, 7-r + 0.5, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1)

# for i in range(10):
#     p1 = [i+1, i+2]
#     p2 = [0, 1]
#     plt.plot(p1, p2, "k-")
#     p2 = [1, 0]
#     plt.plot(p1, p2, "k-")
#     # plt.plot((c, 3-r), (c + 1, 4 - r))
#     # plt.plot((11, 0), (12, 2))
#     print(c, r)
        

for i in range(8):
    plt.axhline(i + 1)

for i in range(8):
    plt.axvline(i + 1)

plt.xlim([0, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([0, 8])
plt.xticks(range(9), range(9))
plt.yticks(range(9), range(9))


plt.show()





