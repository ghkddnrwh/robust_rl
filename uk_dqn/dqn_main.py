# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from dqn_learn import DQNagent
import numpy as np

import os

def main():
    # R = [0, 0.01, 0.02]
    R = [0]

    for r in R:
        simulation_name = "DQN_RL_R=" + str(r)
        env_name = 'CartPole-v1'
        train_num = 5
        total_reward = []    
        total_save_path = os.path.join("test", "iisl2", "dqn_carpole8", env_name, simulation_name)
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
            # pess_env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정

            agent = DQNagent(env)   # A2C 에이전트 객체

        # 학습 진행
            reward = agent.train(500)
            test_reward = agent.test(0)
            total_reward.append(test_reward)
            # agent.save_paremeters(save_path)
            # agent.test()

            reward = np.array(reward)
            print(reward)

        print(total_reward)
if __name__=="__main__":
    main()