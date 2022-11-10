import numpy as np
import math
import matplotlib.pyplot as plt


import os


env_name = 'Pendulum-v1'
total_save_path = os.path.join("data_sac", "pendul", "non_deepcopy", env_name)
data_name = "mass_perturb_test.npy"

R = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
# R = [0, 0.01, 0.02, 0.03, 0.04]
# perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
parameter_perturb_list = [-0.1, -0.07, -0.03, 0, 0.03, 0.07, 0.1]

perturb_type = "Length"

total_reward = np.load(os.path.join(total_save_path, data_name))
print(total_reward.shape)

for r_index, _ in enumerate(R):
    plot_data = total_reward[r_index, :]
    plt.plot(parameter_perturb_list, plot_data, label = "R = %.2f"%R[r_index])


plt.legend()

plt.show()