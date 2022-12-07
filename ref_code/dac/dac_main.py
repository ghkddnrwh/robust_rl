# DQN main
# coded by St.Watermelon

from dac_learn import DQNagent
import gym

def main():
    episode_train_reward = []
    for _ in range(30):
        max_episode_num = 40
        env_name = 'Acrobot-v1'
        env = gym.make(env_name)
        agent = DQNagent(env)

        reward = agent.train(max_episode_num)
        episode_train_reward.append(reward)
    
    print(episode_train_reward)

    # agent.plot_result()

if __name__=="__main__":
    main()