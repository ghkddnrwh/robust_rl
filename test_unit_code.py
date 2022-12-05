import enum
import gym
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import random

import time
<<<<<<< HEAD
import random

# a = [[-87.538, -91.148, -98.826, -101.944, -107.26399999999998, -113.4, -119.95599999999999], [-90.602, -92.186, -96.94, -102.73400000000001, -108.61200000000001, -112.63999999999999, -122.48800000000001], [-87.978, -91.68199999999999, -96.984, -103.16399999999999, -108.054, -113.88799999999999, -119.55800000000002], [-86.548, -91.998, -100.04799999999999, -103.90599999999999, -107.75, -115.0, -119.672], [-85.22999999999999, -91.466, -97.35799999999999, -102.492, -107.026, -113.72, -120.03800000000001], [-89.27600000000001, -93.33599999999998, -95.694, -104.19000000000001, -108.48599999999999, -111.88400000000001, -122.636], [-87.284, -90.704, -99.064, -100.874, -106.91799999999998, -114.734, -122.47999999999999], [-88.35, -94.03599999999999, -94.94000000000001, -99.934, -111.61800000000001, -117.494, -124.77799999999999], [-88.486, -93.39000000000001, -101.52799999999999, -102.854, -110.26399999999998, -120.888, -119.38199999999999], [-90.03999999999999, -96.712, -101.5, -105.306, -112.006, -116.492, -127.62]]
# a = np.array(a)
# plt.plot(a[0], label = "0")
# # plt.plot(a[1], label = "1")
# # plt.plot(a[2], label = "2")
# # plt.plot(a[3], label = "3")
# # plt.plot(a[4], label = "4")
# # plt.plot(a[5], label = "5")
# # plt.plot(a[6], label = "6")
# # plt.plot(a[7], label = "7")
# # plt.plot(a[8], label = "8")
# plt.plot(a[9], label = "9")
# # plt.plot(a[4], label = "4")

# plt.legend()
# plt.show()
=======

# a = [[-85.816, -93.158, -94.09599999999999, -98.52599999999998, -104.61399999999999, -113.45599999999999, -118.83], [-88.44800000000001, -97.25399999999999, -99.056, -101.9, -108.774, -117.5, -121.524], [-92.12799999999999, -96.002, -104.662, -107.79799999999997, -114.622, -123.772, -129.39], [-99.69800000000001, -102.84400000000001, -111.58000000000001, -115.31800000000001, -121.968, -124.09200000000001, -134.12399999999997]]
# a = np.array(a)
# plt.plot(a[0])
# plt.plot(a[1])
# plt.plot(a[2])
# plt.plot(a[3])
# plt.show()

random.seed(0)
env = gym.make("Ant-v4")
state, _ = env.reset(seed = 0)

print(state)

action = env.action_space.sample()
state, _, _, _, _ = env.step(action)
print(action)
print(state)
>>>>>>> ece2ccc4003aac4d2114236289ffc1b29bc57ce1

# zero = 0
# one = 0

# logits = [[10.0, 1.0]]
# logits = np.array(logits)
# for i in range(10000):
#     action = tf.random.categorical(logits, 1)
#     if action == 0:
#         zero+=1
#     else:
#         one+=1

# print(zero / (zero + one))
# print(np.array(action)[0, 0])
# print(action)


random.seed(0)

env = gym.make("Ant-v4")
state, _ = env.reset(seed = 0)

print(state)

action = env.action_space.sample()
state, _, _, _, _ = env.step(action)

print(action)
print(state)

print("Done")

# state_record = []
# pess_state_record = []

# # pess_env = deepcopy(env)
# pess_env = gym.make(env_name)

# env.reset(seed = seed)
# for i in range(10):
#     state, _, _, _, _ = env.step(0)
#     state_record.append(state)

# env.reset(seed = seed)
# for i in range(10):
#     print(state[0])
#     pess_env.reset()
#     exact_state = env.get_exact_state()
#     pess_env.set_state(exact_state)
#     # env.set_state(exact_state)
#     state, _, _, _, _ = env.step(0)
#     exact_state1 = env.get_exact_state()
#     pess_state, _, _, _, _ = pess_env.step(1)
#     exact_state2 = env.get_exact_state()

#     if(exact_state1 != exact_state2):
#         print("Something Wrong")


#     if(state[0] == pess_state[0]):
#         print("Same State")
#     else:
#         print("Different State")


#     pess_state_record.append(state)

# # print(state_record)
# # print(pess_state_record)
# state_record = np.array(state_record)
# pess_state_record = np.array(pess_state_record)
# if(state_record == pess_state_record).all():
#     print("Same")
# else:
#     print("Different")



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
    