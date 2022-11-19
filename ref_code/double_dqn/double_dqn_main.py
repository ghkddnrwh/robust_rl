# DoubleDQN main
# coded by St.Watermelon

from double_dqn_learn import DoubleDQNagent
import gym

def main():

    max_episode_num = 200
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    agent = DoubleDQNagent(env)

    agent.train(max_episode_num)
    test_reward = agent.test(20)
    print(test_reward)

    # agent.plot_result()

if __name__=="__main__":
    main()