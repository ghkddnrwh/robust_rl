# import gym
import numpy as np
import math
import matplotlib.pyplot as plt

import gym
import os

from tabular_robust.robust_q_learn import RobustQAgent

save_path_name = os.path.join("test", "test3")

map_name = "8x8"

slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

slippery_list = [0.66]
r_list = [0]

for r in range(len(r_list)):
    r_boltzmann_reward = []
    r_epsilon_reward = []
    for slip in range(len(slippery_list)):
        q_boltzmann_reward = []
        q_epsilon_reward =[]

        env_name = 'FrozenLake-v1'
        path_env_name = "FrozenLake-v1_slipery=" + str(slippery_list[slip])
        simulation_name = "Robust_RL_R=" + str(r_list[r])

        env_path = os.path.join(save_path_name, path_env_name, simulation_name)
        q_table = np.load(os.path.join(env_path, "q_value.npy"))
        # q_table = q_table[:,0,:,:]
        for q_value in q_table:

            env = gym.make(env_name, map_name = map_name, slippery_value = slippery_list[slip])
            agent = RobustQAgent(env, max_episode_num=1000, r = 0, q_table = q_value)

            boltzmann_reward = agent.test("boltzmann").copy()
            epsilon_reward = agent.test("epsilon_greedy").copy()

            q_boltzmann_reward.append(boltzmann_reward)
            q_epsilon_reward.append(epsilon_reward)

        r_boltzmann_reward.append(np.mean(q_boltzmann_reward))
        r_epsilon_reward.append(np.mean(q_epsilon_reward))

    print("Plot %d"%r)
    plt.subplot(2, 1, 1)
    plt.plot(slippery_list, r_boltzmann_reward, label = "%f"%r_list[r])
    plt.title("Boltzmann Reward")
    plt.subplot(2, 1, 2)
    plt.plot(slippery_list, r_epsilon_reward, label = "%f"%r_list[r])
    plt.title("Epsilon Greedy Reward")


plt.legend()
plt.suptitle("8*8 Map")
plt.show()