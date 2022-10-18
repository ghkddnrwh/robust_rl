# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

r = 0
slippery = 0

action = ["<", "v", ">", "^"]

save_path_name = os.path.join("test", "iisl2", "cliff", "test11")
env_name = 'CliffWalking-v0'
# total_result = np.load(os.path.join("6map_simulation", "total_result.npy"))

# print(total_result)
slippery_list = [0]
r_list = [0]
tau_end_list = [0.002]
reward_list = [1]

tau = tau_end_list[0]
re = reward_list[0]

simulation_name = "Robust_RL_R=" + str(r_list[r])
path_env_name = "CliffWalking-v0_slipery=" + str(slippery_list[slippery])

env_path = os.path.join(save_path_name, path_env_name, simulation_name, "tau=" + str(tau), "re=" + str(re), "q_value.npy")



q_value = np.load(env_path)

q_value = np.mean(q_value, axis = 0)
action = []

for i in q_value:
    ac = i.argmax()
    action.append(ac)

action = np.reshape(action, (4, 12))

for i in action:
    print(i)

# np.savetxt("zz_test/r", q_value)
