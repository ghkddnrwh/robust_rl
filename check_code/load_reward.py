import numpy as np
import math
import matplotlib.pyplot as plt

# import gym
import os


save_simulation = os.path.join("learning_rate", "dense128", "acrobot")
# map_name = "8x8"
data_name = "reward.txt"

if __name__=="__main__":
    # R = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    R = [0, 1]
    for r in R:

        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Acrobot-v1'

        # save_path = os.path.join("data_sac", "pendul", "pess_q_trial2", env_name, simulation_name)
        total_reward = np.loadtxt(os.path.join(save_simulation, env_name, simulation_name, data_name))
        print(total_reward.shape)
        # total_reward = np.mean(total_reward, axis = 0)
        
        plt.plot(total_reward[7, :100])
        # for i in total_reward:
            # plt.plot(i)
        # plt.plot(total_reward, label = "R : %f"%r)
        # plt.plot(total_reward[4])


    plt.legend()
    plt.xlabel("Slippery Miss Probability")
    plt.ylabel("Reward")
    plt.suptitle("8*8 Map Slippery Miss to Reward")
    plt.show()
    plt.legend()