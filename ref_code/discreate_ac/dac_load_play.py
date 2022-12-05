from dac_learn import DQNagent
import gym
import numpy as np

import os

def main():
    R = [0, 0.1, 0.2, 0.3]
    perturb_type = "MASS_POLE"
    train_num = 5

    # perturb_list = [0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # parameter_perturb_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #Gravity
    # parameter_perturb_list = [-0.9, -0.6, -0.3, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #Length
    # parameter_perturb_list = [-0.9, -0.8, -0.6, -0.4, -0.2, 0, 2.0, 4.0, 6.0] #FORCE_MAG
    # parameter_perturb_list = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] #MASS CART
    parameter_perturb_list = [0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0] #MASS POLE
    
    total_reward = []

    for r in R:
        simulation_name = "Robust_RL_R=" + str(r)
        env_name = 'CartPole-v1'

        root_save_path = os.path.join("acd", "tanh", "cartpole", env_name)
        total_save_path = os.path.join(root_save_path, simulation_name)
        perturb_reward = []
        # for perturb in perturb_list:
        for perturb in parameter_perturb_list:
            trial_reward = []
            for train_time in range(train_num):
                save_path = os.path.join(total_save_path, "trial" + str(train_time))
                # env = gym.make(env_name)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
                env = gym.make(env_name, perturb_prob = perturb, perturb_type = perturb_type)  # 환경으로 OpenAI Gym의 pendulum-v0 설정
                agent = DQNagent(env, r)   # A2C 에이전트 객체
                agent.load_weights(save_path)

                # 학습 진행
                # reward = agent.test(perturb)
                reward = agent.test(0)
                trial_reward.append(reward)
            perturb_reward.append(np.mean(trial_reward))
        total_reward.append(perturb_reward)

    print(total_reward)
    np.save(os.path.join(root_save_path, "mass_pole_perturb_test"), np.array(total_reward))

if __name__=="__main__":
    main()