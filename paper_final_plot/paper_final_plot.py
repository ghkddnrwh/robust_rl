import numpy as np
import math
import matplotlib.pyplot as plt

import gym
import os

name_list = ["previous", "attack_q"]

label_list = ["Roubst-Q", "PRQ-Learning"]

map_name = "8x8"
data_name = "local_perturbation_reward_for_paper.npy"

if __name__=="__main__":
    slippery_list = [0]
    # r_list = [0, 0.1, 0.2, 0.3]
    r_list = [0, 0.15, 0.3]
    perturb_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    all_reward = []

    for index, name in enumerate(name_list):
        save_simulation = os.path.join("data", "original_cliff", name)
        total_reward = np.load(os.path.join(save_simulation, data_name))
        
        if index == 0:
            plot_data = total_reward[0, 0, :, 0]
            plt.plot(perturb_list, plot_data, label =  "Q-Learning" + "(R = 0)")

        for r_index in range(total_reward.shape[1]):
            if r_index == 0:
                continue
            plot_data = total_reward[0, r_index, :, 0]
            ref_data = total_reward[0, r_index, 0, 0]
            plt.plot(perturb_list, plot_data, label =  "%s"%label_list[index] + "(R = %.2f)"%r_list[r_index])


    plt.legend()
    plt.xlabel("Action Perturbation Probability")
    plt.ylabel("Accumulated Reward(5 trials mean)")
    # plt.suptitle("CliffWalking")
    # plt.show()
    plt.savefig("Local_Perturbation_CliffWalking", dpi = 400)
    plt.clf()