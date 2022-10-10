from robust_attack_q_learn import RobustQAgent
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

def main(slippery = 0):
    R = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    R = [0]
    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        path_env_name = "FrozenLake-v1_slipery=" + str(slippery)
        env_name = 'FrozenLake-v1'

        save_path = os.path.join("test", "iisl1", "test33", path_env_name, simulation_name)
        try:
            if not(os.path.exists(save_path)):
                os.makedirs(save_path)
            else:
                print("Already Exists Directory")
                return 0    
        except:
            print("Something wrong")
            return 0

        train_num = 1
        max_episode_num = 5000   # 최대 에피소드 설정
        interval = 10           # plot interval

        total_time = []
        total_reward = []
        q_table = []

        for _ in range(train_num):
            env = gym.make(env_name, map_name = "8x8", slippery_value = slippery)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
            agent = RobustQAgent(env, max_episode_num, r)   # A2C 에이전트 객체

        # 학습 진행
            agent.train()

            time, reward = agent.get_average_result(interval)
            q_value = agent.get_q_table()
            
            total_time.append(time)
            total_reward.append(reward)
            q_table.append(q_value)

            boltzmann_reward = agent.test("boltzmann").copy()
            epslion_reward = agent.test("epsilon_greedy").copy()
            print("BOLTZMANN_TEST_REWARD : ", boltzmann_reward)
            print("EPSILON_TEST_REWARD : ", epslion_reward)

        # print(total_reward)
        # mean_total_reward = np.mean(total_reward, axis = 0)

        total_time = np.array(total_time)
        total_reward = np.array(total_reward)
        q_table = np.array(q_table)

        print(q_table)

        np.savetxt(os.path.join(save_path, "time.txt"), total_time)
        np.savetxt(os.path.join(save_path, "reward.txt"), total_reward)
        np.save(os.path.join(save_path, "q_value"), q_table)

    # plt.plot(mean_total_reward)
    # plt.show()

    # agent.plot_result(max_episode_num, interval)

if __name__=="__main__":
    slippery = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
    slippery = [0.66]
    for slip in slippery:
        main(slip)
