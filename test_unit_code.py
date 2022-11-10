import enum
import gym
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

import time


env_name = 'Acrobot-v1'
env = gym.make(env_name, perturb_prob = 0.3, perturb_type = "Length")

# print(env.action_space.sample())
# pess_env = gym.make(env_name)

# state, _ = env.reset()
# pess_env.reset()
# print(state)

# step = 0


# while(True):
#     exact_state = env.get_exact_state()
#     print("Exact State : ",exact_state)
#     pess_env.set_state(exact_state)

#     action = env.action_space.sample()
#     # env.render()
#     next_state, _, done, truncated, _ = env.step(action)
#     pess_next_state, _, _, _, _ = pess_env.step(action)

#     if np.mean((pess_next_state - next_state)**2) < 0.01:
#         print("step : ", step + 1)
#         # print("State : ", next_state)
#     else:
#         print("Not equal")
#         print("State : ", next_state)
#         print("Pess State : ", pess_next_state)
#         while(True):
#             x = 1

#     if done or truncated:
#         print("RESET")
#         state, _ = env.reset()
#         step = 0
#     else:
#         state = next_state
#         step += 1
    # print("Current State : ", state)
    # print("Action : ", action)
    # print("State : ", next_state)
    # print("Pess State : ", pess_next_state)
    # state = next_state
    # step += 1
    # time.sleep(1)

    # break
    