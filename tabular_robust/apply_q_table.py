from robust_q_learn import RobustQAgent
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

np.set_printoptions(suppress=True)

save_simulation = "boltzmann_4map"
map_name = "4x4"

# 시뮬레이션 이름하고 실험하는 환경 변경해야 됨에 유의

def main(slippery, r):
    call_slippery = slippery
    simulation_name = "Robust_RL_R=" + str(r)
    path_env_name = "FrozenLake-v1_slipery=" + str(call_slippery)

    env_name = 'FrozenLake-v1'
    env_slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # env_slippery_list = [0.1]

    call_path = os.path.join(save_simulation, path_env_name, simulation_name)
    q_table = np.load(os.path.join(call_path, "q_value.npy"))
    # print(q_table)

    env_test_reward = []
    for env_slippery in env_slippery_list:
        test_reward = []

        for q_value in q_table:
            env = gym.make(env_name, map_name = map_name, slippery_value = env_slippery)
            agent = RobustQAgent(env, max_episode_num = 1000, r = 0, q_table = q_value)

            agent.test()
            test_reward.append(agent.get_test_reward())
            print("Q_table : ", q_value)
        env_test_reward.append(np.mean(np.array(test_reward)))

    # print(q_table.shape)
    # print(q_table[1])
    return env_test_reward

if __name__=="__main__":
    slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    r_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # slippery_list = [0.1]
    # r_list = [0]

    env_to_env_test_reward = []

    for slippery in slippery_list:
        for r in r_list:
            test_reward = main(slippery, r)
            env_to_env_test_reward.append(test_reward)
            print(test_reward)
    
    env_to_env_test_reward = np.reshape(np.array(env_to_env_test_reward), (len(slippery_list), len(r_list), len(slippery_list)))
    # env_to_env_test_reward = np.reshape(np.array(env_to_env_test_reward), (1, 1, 9))
    print(env_to_env_test_reward)
    np.save(os.path.join(save_simulation, "total_result"), env_to_env_test_reward)