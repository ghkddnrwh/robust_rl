import numpy as np
import math
import matplotlib.pyplot as plt

import gym
import os


save_simulation = os.path.join("data", "previous_robust_rl")
map_name = "8x8"
data_name = "total_reward_for_global_perturbation.npy"

if __name__=="__main__":
    slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
    # slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]
    r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # r_list = [0, 0.05, 0.1, 0.15, 0.2]
    # perturb_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]             # For global perturbation

    # perturb_list = [0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.30]

    # transit_prob_list = [[1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1],
    #                     [0.5, 0.5, 0, 0],
    #                     [0.5, 0, 0.5, 0],
    #                     [0.5, 0, 0, 0.5],
    #                     [0, 0.5, 0.5, 0],
    #                     [0, 0.5, 0, 0.5],
    #                     [0, 0, 0.5, 0.5],
    #                     [0.33, 0.33, 0.33, 0],
    #                     [0.33, 0.33, 0, 0.33],
    #                     [0.33, 0, 0.33, 0.33],
    #                     [0, 0.33, 0.33, 0.33],
    #                     [0.25, 0.25, 0.25, 0.25]
    #                     ]
    transit_prob_list = [None]

    total_reward = np.load(os.path.join(save_simulation, data_name))
    print(total_reward.shape)
    for slip_index in range(len(slippery_list)):
        print(total_reward.shape)

        fig = plt.subplot(3, 4, slip_index + 1)
        for r_index in range(total_reward.shape[1]):
            plot_data = total_reward[slip_index, r_index, :]
            ref_data = total_reward[slip_index, r_index, 0]
            # plot_data -= ref_data
            plt.plot(perturb_list, plot_data, label = "R = %.2f"%r_list[r_index])
            plt.title("Slippery Value : %.2f"%slippery_list[slip_index])
            # plt.xlabel("Perturb Probability")
            # print("----------------")
            # print(total_reward[slip_index, :, :])
            # print("----------------")

    plt.legend()
    plt.xlabel("Perturbation Probability")
    plt.suptitle("8*8 Previous to Global Perturbation (Absolute Reward)")
    plt.show()
    # plt.savefig("image/8*8 Map Slippery Value : %.2f.png"%slippery_list[slip_index], dpi = 200)
    plt.clf()