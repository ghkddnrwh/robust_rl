import numpy as np
import math
import matplotlib.pyplot as plt


import os


env_name = 'Acrobot-v1'
# total_save_path = os.path.join("data_sac", "pendul", "deepcopy_more_trial", env_name)
total_save_path = os.path.join("ac_discrete", "tanh", "acrobot", env_name)
data_name = "action_perturb_test.npy"
data_name = "length_perturb_test.npy"

R = [0, 0.01]

# parameter_perturb_list = [0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # Action perturb
parameter_perturb_list = [-0.6, -0.3, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] # Length perturb
# parameter_perturb_list = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] # Mass perturb
perturb_type = "Action"

total_reward = np.load(os.path.join(total_save_path, data_name))
print(total_reward.shape)

for r_index, _ in enumerate(R):
    plot_data = total_reward[r_index, :]
    plt.plot(parameter_perturb_list, plot_data, label = "R = %.2f"%R[r_index])

plt.xlabel("Action Perturbation Prob")
plt.ylabel("Reward")
plt.legend()

plt.show()