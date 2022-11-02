import gym
from pess_sac_learn import SACagent
import numpy as np

import os
import matplotlib.pyplot as plt


simulation_name = "Robust_RL_R=" + str(r)
env_name = 'Pendulum-v1'
trial_time = 2

total_save_path = os.path.join("data_sac", "pendul", "pess_q_trial6", env_name, simulation_name)
save_path = os.path.join(total_save_path, "trial" + str(train_time))

total_reward = np.load(os.path.join(save_path, "perturb_test"))

R = [0, 0.04]
# perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
perturb_list = [-0.1,0, 0.1]
# parameter_perturb_list = [-0.1, -0.07, -0.03, 0, 0.03, 0.07, 0.1]

for r_index in range(len(R)):
    plot_data = total_reward[r_index, :]
    plt.plot(perturb_list, plot_data, label = "R : %.2f"%R[r_index])

plt.legend()
plt.show()

