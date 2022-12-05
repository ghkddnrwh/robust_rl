# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from attack_ddpg_learn import DDPGagent
import numpy as np
import pickle

import os

def main():
    # R = [0, 0.01]
    # R = [0, 0.01, 0.02, 0.03]
    R = [0]

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Walker2d-v4'
        train_num = 1
        episode_train_reward = []
        episode_test_reward = [] 
        total_save_path = os.path.join("test", "test", "test4", env_name, simulation_name)
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

            env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
            agent = DDPGagent(env, r)   # A2C 에이전트 객체

        # 학습 진행
            reward = agent.train()
            agent.save_paremeters(save_path)
            test_reward = agent.test(0)

            episode_train_reward.append(reward)
            episode_test_reward.append(test_reward)

        print(episode_train_reward)
        print(episode_test_reward)
        with open(os.path.join(total_save_path, "train_reward.pkl") ,'wb') as f:
            pickle.dump(episode_train_reward,f)
        np.savetxt(os.path.join(total_save_path, "reward.txt"), episode_test_reward)

if __name__=="__main__":
    main()