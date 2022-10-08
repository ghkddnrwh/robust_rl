# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

r = 6
slippery = 0

action = ["<", "v", ">", "^"]

save_path_name = "boltzmann_4map"
env_name = 'FrozenLake-v1'
# total_result = np.load(os.path.join("6map_simulation", "total_result.npy"))

# print(total_result)
slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for r in range(len(r_list)):
    R = r_list[r]

    simulation_name = "Robust_RL_R=" + str(R)
    path_env_name = "FrozenLake-v1_slipery=" + str(slippery_list[slippery])

    env_path = os.path.join(save_path_name, path_env_name, simulation_name, "q_value.npy")



    q_value = np.load(env_path)
    q_value = np.mean(q_value, axis = 0)

    print(q_value)
    print(q_value.shape)

# max_value = np.argmax(q_value, axis = 1)
# max_value = np.reshape(max_value, (6, 6))


# for i in range(6):
#     line = "|"
#     for j in range(6):
#         line = line + action[max_value[i, j]] + "|"

#     print(line)
