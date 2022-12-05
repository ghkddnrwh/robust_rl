from dqn_learn import DQNagent
import numpy as np

import os
import matplotlib.pyplot as plt

def main():
    # R = [0, 0.1, 0.2, 0.3]
    R = [0, 0.1, 0.2, 0.3]
    # R = [0, 0.01, 0.02, 0.03]
    # perturb_type = "MASS_POLE"
    train_num = 5

    # parameter_perturb_list = [0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # parameter_perturb_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #Gravity
    # parameter_perturb_list = [-0.9, -0.6, -0.3, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #Length
    # parameter_perturb_list = [-0.9, -0.8, -0.6, -0.4, -0.2, 0, 2.0, 4.0, 6.0] #FORCE_MAG
    # parameter_perturb_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #MASS CART
    # parameter_perturb_list = [0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0] #MASS POLE

    # Acrobot
    perturb_list = [0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3]
    
    total_reward = []


    # simulation_name = "Robust_RL_R=" + str(r)
    env_name = 'Acrobot-v1'

    root_save_path = os.path.join("ac_discrete", "tanh", "acrobot", env_name)

    data_name = "action_perturb_test.npy"

    reward = np.load(os.path.join(root_save_path, data_name))

    for r in range(len(R)):
        plt.plot(perturb_list, reward[r], label = "R:%.2f"%R[r])

    plt.legend()
    plt.xlabel("Action Perturb Prob")
    plt.show()



if __name__=="__main__":
    main()