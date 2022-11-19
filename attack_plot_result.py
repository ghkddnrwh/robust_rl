import numpy as np
import math
import matplotlib.pyplot as plt


import os


env_name = 'Pendulum-v1'
# total_save_path = os.path.join("data_sac", "pendul", "deepcopy_more_trial", env_name)
total_save_path = os.path.join("data_ddpg", "pendul", "deepcopy_more_trial", env_name)
data_name = "length_perturb_test5.npy"

R = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
R = [0, 0.01]
# perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
# parameter_perturb_list = [-0.1, -0.07, -0.03, 0, 0.03, 0.07, 0.1]
parameter_perturb_list = [-0.6,-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3] #2
# parameter_perturb_list = perturb_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4] # action perturb
# parameter_perturb_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] # mass
parameter_perturb_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] # gravity4 length4
parameter_perturb_list = [-0.9, -0.8, -0.7]
perturb_type = "Gravity"

total_reward = np.load(os.path.join(total_save_path, data_name))
print(total_reward.shape)

for r_index, _ in enumerate(R):
    plot_data = total_reward[r_index, :]
    plt.plot(parameter_perturb_list, plot_data, label = "R = %.2f"%R[r_index])

plt.xlabel("Length Perturbation Prob")
plt.ylabel("Reward")
plt.legend()

plt.show()