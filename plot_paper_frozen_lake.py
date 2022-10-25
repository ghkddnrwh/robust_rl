import numpy as np
import math
import matplotlib.pyplot as plt

import gym
import os

name_list = ["previous", "attack_q2"]

label_list = ["Roubst-Q", "PRQ-Learning"]

map_name = "8x8"
# data_name = "global_perturbation_for_paper.npy"
data_name = "local_perturbation_reward_for_test.npy"

if __name__=="__main__":
    slippery_index = 0
    r_list = [0, 0.15, 0.3]
    # r_list = [0.15, 0.3]
    perturb_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # perturb_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    all_reward = []
    fig, ax = plt.subplots(figsize = (3, 2.5))
    for index, name in enumerate(name_list):
        save_simulation = os.path.join("paper_final", "frozen_lake", name)
        total_reward = np.load(os.path.join(save_simulation, data_name))
        
        if index == 0:
            plot_data = total_reward[slippery_index, 0, :, 0]
            plt.plot(perturb_list, plot_data, label =  "Q-Learning" + "(R = 0)")

        for r_index in range(total_reward.shape[1]):
            if r_index == 0:
                continue
            plot_data = total_reward[slippery_index, r_index, :, 0]
            # ref_data = total_reward[slippery_index, r_index, 0, 0]
            plt.plot(perturb_list, plot_data, label =  "%s"%label_list[index] + "(R = %.2f)"%r_list[r_index])

    # plt.legend()
    plt.xlabel("Action Perturbation Probability")
    plt.ylabel("Accumulated Reward")
    # plt.suptitle("Frozen-Lake(Slippery Probability = 0)")
    fig.tight_layout()
    # plt.show()
    plt.savefig("FrozenLake_Slippery=0", dpi = 400)
    plt.clf()