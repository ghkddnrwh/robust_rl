import enum
import gym
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

import time


env_name = 'CartPole-v1'
env = gym.make(env_name)
seed = 1

state_record = []
pess_state_record = []

# pess_env = deepcopy(env)
pess_env = gym.make(env_name)

env.reset(seed = seed)
for i in range(10):
    state, _, _, _, _ = env.step(0)
    state_record.append(state)

env.reset(seed = seed)
for i in range(10):
    print(state[0])
    pess_env.reset()
    exact_state = env.get_exact_state()
    pess_env.set_state(exact_state)
    # env.set_state(exact_state)
    state, _, _, _, _ = env.step(0)
    exact_state1 = env.get_exact_state()
    pess_state, _, _, _, _ = pess_env.step(1)
    exact_state2 = env.get_exact_state()

    if(exact_state1 != exact_state2):
        print("Something Wrong")


    if(state[0] == pess_state[0]):
        print("Same State")
    else:
        print("Different State")


    pess_state_record.append(state)

# print(state_record)
# print(pess_state_record)
state_record = np.array(state_record)
pess_state_record = np.array(pess_state_record)
if(state_record == pess_state_record).all():
    print("Same")
else:
    print("Different")



# state_same_count = [0, 0, 0, 0, 0, 0]
# same_count = 0
# reward_same_count = 0


# for i in range(10000):
#     env.reset(seed = i)
#     env.step(1)
#     env.step(1)
#     env.step(1)
#     next_state, reward, _, _, _ = env.step(0)
#     env.reset(seed = i)
#     env.step(1)
#     env.step(1)
#     env.step(1)
#     new_next_state, new_reward, _, _, _ = env.step(1)

#     if(next_state == new_next_state).all():
#         same_count += 1
#     if next_state[0] == new_next_state[0]:
#         state_same_count[0] += 1
#     if next_state[1] == new_next_state[1]:
#         state_same_count[1] += 1
#     if next_state[2] == new_next_state[2]:
#         state_same_count[2] += 1
#     if next_state[3] == new_next_state[3]:
#         state_same_count[3] += 1
#     # if next_state[4] == new_next_state[4]:
#     #     state_same_count[4] += 1
#     # if next_state[5] == new_next_state[5]:
#     #     state_same_count[5] += 1
#     if reward == new_reward:
#         reward_same_count += 1

# print(state_same_count)
# print(same_count)
# print(reward_same_count)






# env.seed(0)
# env.reset(seed = 0)

# print(env.get_exact_state())


# print(env.get_exact_state())

# print(next_state)

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
    