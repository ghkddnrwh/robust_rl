# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from dqn_learn import DQNagent
import numpy as np

import os

def main():
    # R = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # R = [0]
    # perturb_list = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
    # parameter_perturb_list = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    # parameter_perturb_list = [0]
    perturb_type = "Length"
    # R = [0.1]
    total_reward = []
    train_num = 1

    env_name = 'Acrobot-v1'

    # root_save_path = os.path.join("test", "iisl2", "acrobot", env_name)
    total_save_path = os.path.join("test", "iisl2", "dqn", "acrobot11", env_name)
    trial_reward = []
    for train_time in range(train_num):
        save_path = os.path.join(total_save_path, "trial" + str(train_time))
        env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
        agent = DQNagent(env)   # A2C 에이전트 객체
        agent.load_weights(save_path)

        # 학습 진행
        # reward = agent.test(perturb)
        # test_reward = agent.test(deterministic=True)
        # test_reward1 = agent.test(deterministic=False)
        train_reward = agent.train_for_test(10)
        agent.load_weights(save_path)
        reward = agent.test(deterministic=True)
        reward1 = agent.test(deterministic=False)
        # reward1 = agent.test_pess_action(0)
        # trial_reward.append(reward)

    # print(test_reward)
    # print(test_reward1)
    print(train_reward)
    print(reward)
    print(reward1)

if __name__=="__main__":
    main()