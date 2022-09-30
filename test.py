import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import os

total_result = np.load(os.path.join("boltzmann_6map", "total_result.npy"))

print(total_result)
slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]



# for slip in range(total_result.shape[0]):
#     for R in range(total_result.shape[1]):
#         # print("hello")
#         # print(total_result[:, R ,:])
#         env_to_env = total_result[slip, R, :]
#         # print(env_to_env)
#         # np.fill_diagonal(env_to_env, val = 0)
#         # result = np.mean(env_to_env, axis = 0)
#         # if i == 0 or i == 6:
#         plt.subplot(3, 3, slip + 1)
#         plt.plot(slippery_list, env_to_env, label = str(r_list[R]))
#         plt.title("Train Env SLippery %.1f"%slippery_list[slip])
#         plt.xlabel("Test Env Slippery")
#         # i = i + 1


for R in range(total_result.shape[1]):
    env_to_env = total_result[:, R, :]
    # np.fill_diagonal(env_to_env, val = 0)
    result = np.mean(env_to_env, axis = 0)
    # if i == 0 or i == 6:

    plt.plot(slippery_list, result, label = str(r_list[R]))
    # i = i + 1 

plt.xlabel("Test Env Slippery")
plt.title("Average")
plt.legend()
plt.show()