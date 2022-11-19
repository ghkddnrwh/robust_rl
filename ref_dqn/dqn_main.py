from dqn_learn import DQNagent
import gym
import numpy as np

import os


def main():
    # simulation_name = "Robust_RL_R=" + str(r)
    env_name = 'Acrobot-v1'
    train_num = 1
    total_reward = []    
    total_test_reward = []
    total_save_path = os.path.join("test", "iisl2", "dqn", "acrobot20", env_name)
    for train_time in range(train_num):
        save_path = os.path.join(total_save_path, "trial" + str(train_time))
        try:
            if not(os.path.exists(save_path)):
                os.makedirs(save_path)
            else:
                print("Already Exists Directory")
                return 0    
        except:
            print("Something wrong")
            return 0
        max_episode_num = 100
        env_name = 'Acrobot-v1'
        env = gym.make(env_name)
        agent = DQNagent(env)

        reward = agent.train(max_episode_num)
        agent.save_paremeters(save_path)
        # train_reward = agent.train_for_test(10)
        test_reward = agent.test()
        test_reward2 = agent.test(deterministic=False)

        # agent.plot_result()
        print(reward)
        print(train_reward)
        print(test_reward)
        print(test_reward2)

if __name__=="__main__":
    main()