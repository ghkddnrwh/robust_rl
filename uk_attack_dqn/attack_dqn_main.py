# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from attack_dqn_learn import DQNAgent
import numpy as np

import os

def main():
    # R = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    R = [0]

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Acrobot-v1'
        train_num = 1
        total_reward = []    
        total_save_path = os.path.join("test", "new_test", "acrobot2", env_name, simulation_name)
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
            pess_env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정

            agent = DQNAgent(env, pess_env, r)   # A2C 에이전트 객체

        # 학습 진행
            reward = agent.train()
            agent.save_paremeters(save_path)
            agent.test()

            reward = np.array(reward)
            total_reward.append(reward)

        min_len = 0
        for re in total_reward:
            min_len = min(min_len, len(re))

        np_total_reward = []
        for re in total_reward:
            np_total_reward.append(re[:min_len])
        total_reward = np.array(np_total_reward)
        np.savetxt(os.path.join(total_save_path, "reward.txt"), np_total_reward)

if __name__=="__main__":
    main()