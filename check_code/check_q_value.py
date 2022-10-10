# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

r = 0
slippery = 0

action = ["<", "v", ">", "^"]

save_path_name = os.path.join("data", "attack_q", "boltzmann_8map")
env_name = 'FrozenLake-v1'
# total_result = np.load(os.path.join("6map_simulation", "total_result.npy"))

# print(total_result)
slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


simulation_name = "Robust_RL_R=" + str(r_list[r])
path_env_name = "FrozenLake-v1_slipery=" + str(slippery_list[slippery])

env_path = os.path.join(save_path_name, path_env_name, simulation_name, "q_value.npy")



q_value = np.load(env_path)

print(q_value.shape)

print(q_value[0, 0, :, :])
print(q_value[0, 1, :, :])
