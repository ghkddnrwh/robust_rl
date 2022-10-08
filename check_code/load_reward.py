import numpy as np
import math
import matplotlib.pyplot as plt

import gym
import os


save_simulation = "data/boltzmann_8map"
map_name = "8x8"
data_name = "slippery_miss_total_reward.npy"

if __name__=="__main__":
    slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
    r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # perturb_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    perturb_list = [-0.1, -0.07, -0.03, 0, 0.03, 0.07, 0.1]

    total_reward = np.load(os.path.join(save_simulation, data_name))

    for slip_index in range(total_reward.shape[0]):
        plt.subplot(3, 4, slip_index + 1)
        for r_index in range(total_reward.shape[1]):
            plt.plot(perturb_list, total_reward[slip_index, r_index, :] - total_reward[slip_index, r_index, 3], label = "R = %.2f"%r_list[r_index])
        plt.title("Slippery : %.2f"%slippery_list[slip_index])
        # plt.xlabel("Perturb Probability")
        # print("----------------")
        # print(total_reward[slip_index, :, :])
        # print("----------------")

    plt.legend()
    plt.xlabel("Slippery Miss Probability")
    plt.ylabel("Reward")
    plt.suptitle("8*8 Map Slippery Miss to Reward")
    plt.show()
    plt.legend()