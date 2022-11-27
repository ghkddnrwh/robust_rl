# DQN main
# coded by St.Watermelon

from dqn_learn import DQNagent
import gym

def main():
    episode_train_reward = []
    episode_test_reward = []
    for i in range(3):
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
        agent = DQNagent(env)

        train_reward = agent.train()
        test_reward = agent.test(10)

        episode_train_reward.append(train_reward)
        episode_test_reward.append(test_reward)

    print(episode_train_reward)
    print(episode_test_reward)

    # agent.plot_result()

if __name__=="__main__":
    main()