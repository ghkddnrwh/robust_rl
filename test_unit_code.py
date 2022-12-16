import enum
import gym
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import random

import time

a = [[-500.0, -500.0, -500.0, -421.0, -272.0, -236.0, -260.0, -257.0, -330.0, -271.0, -378.0, -335.0, -289.0, -214.0, -182.0, -269.0, -455.0, -280.0, -253.0, -300.0, -247.0, -215.0, -271.0, -220.0, -232.0, -297.0, -269.0, -210.0, -291.0, -500.0, -350.0, -309.0, -359.0, -346.0, -407.0, -300.0, -258.0, -283.0, -218.0, -236.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -388.0, -372.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -412.0, -500.0, -284.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -378.0, -357.0, -360.0, -333.0, -282.0], [-500.0, -411.0, -500.0, -500.0, -500.0, -500.0, -455.0, -500.0, -432.0, -344.0, -393.0, -500.0, -344.0, -387.0, -484.0, -464.0, -500.0, -395.0, -259.0, -243.0, -272.0, -161.0, -210.0, -203.0, -242.0, -233.0, -200.0, -237.0, -214.0, -149.0, -227.0, -220.0, -225.0, -156.0, -155.0, -177.0, -134.0, -212.0, -171.0, -112.0], [-312.0, -280.0, -500.0, -500.0, -500.0, -500.0, -434.0, -300.0, -310.0, -276.0, -226.0, -500.0, -294.0, -294.0, -386.0, -353.0, -291.0, -299.0, -420.0, -255.0, -308.0, -500.0, -212.0, -292.0, -191.0, -165.0, -218.0, -196.0, -138.0, -197.0, -224.0, -157.0, -186.0, -139.0, -184.0, -128.0, -175.0, -154.0, -153.0, -128.0], [-213.0, -214.0, -265.0, -236.0, -234.0, -317.0, -318.0, -168.0, -236.0, -222.0, -154.0, -150.0, -147.0, -182.0, -310.0, -210.0, -229.0, -151.0, -164.0, -178.0, -122.0, -197.0, -252.0, -219.0, -170.0, -245.0, -165.0, -182.0, -208.0, -179.0, -128.0, -147.0, -164.0, -166.0, -176.0, -148.0, -131.0, -154.0, -163.0, -151.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -348.0, -346.0, -295.0, -323.0, -342.0, -226.0, -377.0, -500.0, -500.0, -500.0, -332.0, -267.0, -230.0, -268.0, -276.0, -264.0, -223.0, -322.0, -244.0, -234.0, -211.0, -175.0, -158.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -452.0, -500.0, -500.0, -500.0, -432.0, -362.0, -318.0, -365.0, -500.0, -384.0, -500.0, -500.0, -500.0, -474.0, -242.0, -272.0, -484.0, -271.0, -364.0, -177.0, -420.0, -338.0, -500.0, -440.0, -500.0, -474.0, -403.0, -279.0, -385.0], [-500.0, -500.0, -500.0, -335.0, -353.0, -265.0, -314.0, -341.0, -260.0, -308.0, -500.0, -500.0, -249.0, -270.0, -407.0, -291.0, -439.0, -266.0, -269.0, -225.0, -280.0, -383.0, -285.0, -212.0, -279.0, -440.0, -390.0, -470.0, -363.0, -500.0, -306.0, -333.0, -490.0, -338.0, -237.0, -308.0, -225.0, -293.0, -181.0, -298.0], [-300.0, -500.0, -500.0, -500.0, -314.0, -374.0, -277.0, -193.0, -172.0, -330.0, -278.0, -261.0, -216.0, -256.0, -243.0, -275.0, -343.0, -335.0, -336.0, -182.0, -265.0, -212.0, -240.0, -260.0, -251.0, -213.0, -324.0, -177.0, -185.0, -158.0, -181.0, -196.0, -199.0, -220.0, -147.0, -191.0, -158.0, -165.0, -181.0, -141.0], [-291.0, -375.0, -335.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -397.0, -500.0], [-348.0, -500.0, -500.0, -500.0, -500.0, -398.0, -500.0, -500.0, -500.0, -500.0, -418.0, -500.0, -492.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -461.0, -338.0, -293.0, -191.0, -170.0, -353.0, -250.0, -162.0, -260.0, -197.0, -256.0, -165.0, -195.0, -171.0, -268.0, -210.0, -181.0, -131.0, -197.0, -211.0, -161.0], [-418.0, -219.0, -417.0, -266.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -373.0, -410.0, -303.0, -425.0, -500.0, -456.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -440.0, -442.0, -372.0, -298.0, -317.0, -141.0, -224.0, -162.0, -222.0, -148.0, -259.0, -296.0, -156.0, -140.0, -172.0, -107.0, -169.0], [-231.0, -285.0, -437.0, -207.0, -306.0, -306.0, -274.0, -500.0, -500.0, -500.0, -318.0, -361.0, -500.0, -474.0, -259.0, -253.0, -381.0, -252.0, -281.0, -462.0, -265.0, -500.0, -158.0, -169.0, -191.0, -203.0, -191.0, -186.0, -253.0, -205.0, -184.0, -223.0, -201.0, -183.0, -185.0, -211.0, -191.0, -132.0, -249.0, -262.0], [-500.0, -500.0, -500.0, -500.0, -419.0, -265.0, -430.0, -500.0, -306.0, -183.0, -297.0, -276.0, -194.0, -500.0, -475.0, -317.0, -303.0, -339.0, -500.0, -238.0, -406.0, -500.0, -322.0, -414.0, -406.0, -500.0, -473.0, -500.0, -500.0, -381.0, -262.0, -293.0, -359.0, -390.0, -500.0, -500.0, -295.0, -356.0, -238.0, -311.0], [-408.0, -465.0, -394.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -415.0, -500.0, -241.0, -379.0, -229.0, -224.0, -249.0, -166.0, -317.0, -223.0, -236.0, -327.0, -276.0, -287.0, -246.0, -325.0, -400.0, -265.0, -459.0, -415.0, -353.0, -274.0, -500.0, -297.0, -301.0, -444.0], [-500.0, -500.0, -500.0, -500.0, -498.0, -452.0, -423.0, -389.0, -500.0, -500.0, -500.0, -500.0, -390.0, -368.0, -325.0, -335.0, -298.0, -260.0, -458.0, -156.0, -291.0, -365.0, -313.0, -292.0, -175.0, -238.0, -240.0, -325.0, -399.0, -244.0, -331.0, -463.0, -199.0, -208.0, -242.0, -223.0, -252.0, -247.0, -270.0, -243.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -359.0, -413.0, -446.0, -308.0, -387.0, -297.0, -225.0, -159.0, -139.0, -196.0, -193.0, -182.0, -215.0, -190.0, -191.0, -202.0, -203.0, -214.0, -162.0, -216.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-236.0, -203.0, -323.0, -251.0, -278.0, -474.0, -440.0, -281.0, -377.0, -262.0, -375.0, -423.0, -259.0, -363.0, -246.0, -500.0, -429.0, -233.0, -259.0, -417.0, -334.0, -205.0, -219.0, -227.0, -172.0, -197.0, -143.0, -190.0, -222.0, -301.0, -156.0, -123.0, -178.0, -147.0, -167.0, -130.0, -235.0, -197.0, -190.0, -225.0], [-500.0, -363.0, -411.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -463.0, -346.0, -316.0, -385.0, -257.0, -377.0, -308.0, -336.0, -390.0, -312.0, -362.0, -295.0, -364.0, -404.0, -401.0, -304.0, -318.0, -232.0, -288.0, -278.0, -285.0, -275.0, -262.0, -238.0, -210.0, -259.0, -157.0, -349.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -498.0, -483.0, -500.0, -296.0, -339.0, -254.0, -327.0, -294.0, -194.0, -220.0, -247.0, -233.0, -204.0, -191.0, -229.0, -224.0, -253.0, -208.0, -166.0, -271.0], [-394.0, -500.0, -473.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -397.0, -500.0, -500.0, -500.0, -500.0, -500.0, -443.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -459.0, -406.0, -281.0, -403.0, -334.0, -242.0, -263.0, -268.0, -242.0, -292.0, -450.0, -208.0], [-448.0, -500.0, -383.0, -249.0, -474.0, -200.0, -350.0, -500.0, -372.0, -261.0, -200.0, -192.0, -224.0, -223.0, -183.0, -196.0, -198.0, -140.0, -201.0, -140.0, -190.0, -151.0, -177.0, -221.0, -273.0, -307.0, -217.0, -175.0, -207.0, -231.0, -152.0, -183.0, -263.0, -153.0, -185.0, -187.0, -151.0, -119.0, -146.0, -189.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -444.0, -496.0, -479.0, -321.0, -287.0, -234.0, -263.0, -234.0, -219.0, -276.0, -418.0, -229.0, -243.0, -228.0, -253.0, -214.0, -321.0, -249.0, -176.0, -293.0, -217.0, -206.0, -179.0, -183.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -487.0, -197.0, -189.0, -282.0, -201.0, -318.0, -486.0, -193.0, -230.0, -184.0, -278.0, -217.0, -197.0, -206.0, -219.0, -186.0, -191.0, -151.0, -154.0, -142.0, -199.0, -164.0, -197.0, -279.0, -190.0, -206.0, -117.0, -169.0, -127.0, -193.0, -199.0, -186.0, -173.0, -194.0, -145.0]]
a = np.array(a)
plt.plot(a[0+10])
plt.plot(a[1+10])
plt.plot(a[2+10])
plt.plot(a[3+10])
plt.plot(a[4+10])
plt.show()

# random.seed(0)
# env = gym.make("Ant-v4")
# state, _ = env.reset(seed = 0)

# print(state)

# action = env.action_space.sample()
# state, _, _, _, _ = env.step(action)
# print(action)
# print(state)

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
    