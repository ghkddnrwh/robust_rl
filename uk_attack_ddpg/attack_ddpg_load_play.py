# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from attack_ddpg_learn import DDPGagent
import numpy as np

import os

def main():
    R = [0, 0.01]
    # R = [0]
    # perturb_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4]
    parameter_perturb_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    parameter_perturb_list = [-0.9, -0.8, -0.7]
    # parameter_perturb_list = [0]
    perturb_type = "Length"
    # R = [0.1]
    total_reward = []
    train_num = 5

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'Pendulum-v1'

        root_save_path = os.path.join("data_ddpg", "pendul", "deepcopy_more_trial", env_name)
        total_save_path = os.path.join(root_save_path, simulation_name)
        perturb_reward = []
        # for perturb in perturb_list:
        for perturb in parameter_perturb_list:
            trial_reward = []
            for train_time in range(train_num):
                save_path = os.path.join(total_save_path, "trial" + str(train_time))
                # env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
                env = gym.make(env_name, perturb_prob = perturb, perturb_type = perturb_type)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
                agent = DDPGagent(env, r)   # A2C 에이전트 객체
                agent.load_weights(save_path)

                # 학습 진행
                # reward = agent.test(perturb)
                reward = agent.test(0)
                trial_reward.append(reward)
            perturb_reward.append(np.mean(trial_reward))
        total_reward.append(perturb_reward)

    print(total_reward)
    np.save(os.path.join(root_save_path, "length_perturb_test"), np.array(total_reward))

if __name__=="__main__":
    main()