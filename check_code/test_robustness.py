from tabular_robust.robust_q_learn import RobustQAgent
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

save_simulation = os.path.join("data", "taxi", "robust_q", "epsilong_greedy")
map_name = "8x8"

def main(slippery, r_list, perturb_list, transit_prob_list):
    env_test_reward = []
    for r in r_list:
        simulation_name = "Robust_RL_R=" + str(r)
        path_env_name = "Taxi-v3_slipery=" + str(slippery)

        env_name = 'Taxi-v3'

        call_path = os.path.join(save_simulation, path_env_name, simulation_name)
        q_table = np.load(os.path.join(call_path, "q_value.npy"))
        # q_table = q_table[:,0,:,:]
        # print(q_table.shape)
        # break
        
        perturb_reward = []
        for perturb in perturb_list:
            transit_prob_reward = []
            for transit_prob in transit_prob_list:
                test_reward = []
                for q_value in q_table:
                    env = gym.make(env_name, perturb = perturb)
                    agent = RobustQAgent(env, max_episode_num = 1000, r = 0, q_table = q_value)

                    agent.test("epsilon_greedy")
                    test_reward.append(agent.get_test_reward())

                    print("Reward : ", test_reward)
                transit_prob_reward.append(np.mean(np.array(test_reward)))
            perturb_reward.append(transit_prob_reward)
        env_test_reward.append(perturb_reward)
    return env_test_reward

if __name__=="__main__":
    # slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
    slippery_list = [0]
    # slippery_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]
    r_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    # r_list = [0, 0.05, 0.1, 0.15]
    # perturb_list = [0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2]
    # perturb_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    # perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]             # For global perturbation
    transit_prob_list = [None]
    # transit_prob_list = [[1, 0, 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [0, 0, 0, 1],
    #                     [0.5, 0.5, 0, 0],
    #                     [0.5, 0, 0.5, 0],
    #                     [0.5, 0, 0, 0.5],
    #                     [0, 0.5, 0.5, 0],
    #                     [0, 0.5, 0, 0.5],
    #                     [0, 0, 0.5, 0.5],
    #                     [0.33, 0.33, 0.33, 0],
    #                     [0.33, 0.33, 0, 0.33],
    #                     [0.33, 0, 0.33, 0.33],
    #                     [0, 0.33, 0.33, 0.33],
    #                     [0.25, 0.25, 0.25, 0.25]
    #                     ]

    # For Test
    # slippery_list = [0.8, 0.7]
    # r_list = [0, 0.05, 0.1]
    # perturb_list = [0]

    env_to_env_test_reward = []

    for slippery in slippery_list:
        test_reward = main(slippery, r_list, perturb_list, transit_prob_list)
        env_to_env_test_reward.append(test_reward)
    env_to_env_test_reward = np.reshape(np.array(env_to_env_test_reward), (len(slippery_list), len(r_list), len(perturb_list), len(transit_prob_list)))

    np.save(os.path.join(save_simulation, "total_reward_for_local_perturbation"), env_to_env_test_reward)