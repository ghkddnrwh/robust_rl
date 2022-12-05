# DQN main
# coded by St.Watermelon

from dqn_learn import DQNagent
import numpy as np
import pickle
import gym
import os

def main():
    R = [0, 0.1, 0.2]
    # R = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    # R = [0.07, 0.08, 0.09]
    # R = [0]

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Acrobot-v1'

        train_num = 5
        episode_train_reward = []
        episode_test_reward = []
        total_save_path = os.path.join("ac_discrete", "relu", "acrobot", env_name, simulation_name)

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

            env = gym.make(env_name)
            agent = DQNagent(env, r)

            train_reward = agent.train()
            agent.save_paremeters(save_path)
            test_reward = agent.test()

            episode_train_reward.append(train_reward)
            episode_test_reward.append(test_reward)

        print(episode_train_reward)
        print(episode_test_reward)
        with open(os.path.join(total_save_path, "train_reward.pkl") ,'wb') as f:
            pickle.dump(episode_train_reward,f)
        np.savetxt(os.path.join(total_save_path, "test_reward.txt"), np.array(episode_test_reward))

if __name__=="__main__":
    main()


