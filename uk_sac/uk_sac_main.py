# SAC main (tf2 subclassing API version)
# coded by St.Watermelon
## SAC 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym
from attack_robust_sac_learn import SACagent

def main():
    env = gym.make("Pendulum-v1")
    # state = env.reset()
    # print("Init state : ", len(state))
    agent = SACagent(env)  # SAC 에이전트 객체

    # 학습 진행
    total_reward = agent.train()

    print(total_reward)

    # 학습 결과 도시
    # agent.plot_result()

if __name__=="__main__":
    main()