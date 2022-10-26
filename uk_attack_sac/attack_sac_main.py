# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from pess_sac_learn import SACagent
import numpy as np

import os

def main():
    R = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    # R = [0.1]

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Pendulum-v1'

        save_path = os.path.join("data_sac", "pendul", "pess_q_trial4", env_name, simulation_name)
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
        # max_episode_num = 5000   # 최대 에피소드 설정
        # interval = 10           # plot interval

        # total_time = []
        total_reward = []
        # q_table = []

        for _ in range(train_num):
            env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
            pess_env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정

            agent = SACagent(env, pess_env, r)   # A2C 에이전트 객체

        # 학습 진행
            reward = agent.train()
            agent.save_paremeters(save_path)

            total_reward.append(reward)

        total_reward = np.array(total_reward)

        np.savetxt(os.path.join(save_path, "reward.txt"), total_reward)

if __name__=="__main__":
    main()