# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

r = 2
slippery = 5

save_path_name = "boltzmann_6map"
env_name = 'FrozenLake-v1'
# total_result = np.load(os.path.join("6map_simulation", "total_result.npy"))

# print(total_result)
slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # r_reward = []
for slip in range(len(slippery_list)):
    for r in range(len(r_list)):
        R = r_list[r]

        simulation_name = "Robust_RL_R=" + str(R)
        path_env_name = "FrozenLake-v1_slipery=" + str(slippery_list[slip])

        env_path = os.path.join(save_path_name, path_env_name, simulation_name, "reward.txt")

        reward = np.loadtxt(env_path)

        reward = np.mean(reward, axis = 0)
        # r_reward.append(reward)
        # reward = np.reshape(reward, (100, 3))
        # reward = np.mean(reward, axis = 1)

        plt.subplot(3, 3, slip + 1)
        plt.plot(reward, label = "R : %.2f"%(R))
        plt.title("Slip value : %.1f"%slippery_list[slip])
    # r_reward = np.reshape(r_reward, (9, 100))
    # r_reward= np.mean(r_reward, axis = 0)
    # plt.plot(r_reward, label = "%f"%(R))

plt.legend()
plt.suptitle("6*6 Map")
plt.show()