from cmath import isnan

import numpy as np
import matplotlib.pyplot as plt
import math
import random

from tabular_robust.replaybuffer import ReplayBuffer

EPS_START = 1
EPS_END = 0.01

TAU_START = 1
TAU_END = 0.01


class TabularQ(object):
    def __init__(self, state_kind, action_kind):
        self.state_kind = state_kind
        self.action_kind = action_kind
        self.q_table = np.zeros((state_kind, action_kind), dtype = np.float32)
        self.v_table = np.zeros((state_kind), dtype = np.float32)

    # state, action에 대한 Q-Value 값 리턴
    def __call__(self, state, action):
        if state < 0 or state >= self.state_kind:
            print("Wrong state position")
            return 0
        if action < 0 or action >= self.action_kind:
            print("Wrong action position")
            return 0
            
        return self.q_table[state, action]

    # state에서 모든 action에 대한 Q-value 값을 벡터로 리턴
    def get_v_values(self, state):
        if state < 0 or state >= self.state_kind:
            print("Wrong state position")
            return 0

        return self.q_table[state].copy()

    def get_v_value(self):
        return self.v_table.copy()

    def get_q_value(self):
        return self.q_table.copy()

    # Q-value, V-value 업데이트 이때 Greedy policy를 가정하기 때문에 V-value는 max로 업데이트
    # 여기서 V-value 업데이트 한는 값을 behavior policy로 해야되나 아니면 Greedy로 해야 되나?
    def update_value_function(self, state, action, update_value, mode = "boltzmann", epsilon = TAU_END):
        if state < 0 or state >= self.state_kind:
            print("Wrong state position")
            return 0
        if action < 0 or action >= self.action_kind:
            print("Wrong action position")
            return 0
        self.q_table[state, action] = update_value
        if mode == "greedy":
            self.v_table[state] = max(self.v_table[state], update_value)
        elif mode == "boltzmann":
            state_values = np.array(self.q_table[state].copy(), dtype = np.float64)
            state_probability = np.exp(state_values / epsilon) / np.sum(np.exp(state_values / epsilon))
            for prob in state_probability:
                if math.isnan(prob):
                    print("There is nan in action value")
                    print("State Values ", state_values)
                    while(True):
                        x = 1
            
            self.v_table[state] = np.dot(state_values, state_probability)

    # 학습된 에이전트로 가져오기
    def push_q_table(self, q_table):
        self.q_table = q_table.copy()
        self.v_table = self.q_table.max(axis = 1)

    def print(self):
        print("--------------------")
        for state in range(self.state_kind):
            print(self.q_table[state])
        print("--------------------")


class RobustQAgent(object):
    def __init__(self, env, max_episode_num, r = 0, q_table = None, tau = 0.002, re = 20):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.ALPHA = 0.01                   # Update 비율 / 여기서는 신경망을 사용 안하기 때문에 learning rate 대신 존재
        self.R = r                        # robustness 의 정도 / 클수록 robustness 증가
        self.LEARNING_AFTER_STEP = 1000
        self.PTM_STEP = 10000               # PTM 업데이트 스텝
        self.MAX_EPISODE_NUM = max_episode_num

        self.TEST_STEP = 1000

        self.tau = tau
        self.re = re

        self.env = env
        self.state_kind = env.observation_space.n
        self.action_kind = env.action_space.n

        self.robust_q = TabularQ(self.state_kind, self.action_kind)
        if q_table is not None:
            self.robust_q.push_q_table(q_table)
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)
        self.PTM = np.zeros((self.state_kind, self.state_kind))     # state to state transition의 가능성을 판단하는 행렬
        self.PTM_for_previous = np.zeros((self.state_kind))         # 이전 Robust RL 구현을 위해 추가한 vector

        self.save_epi_time = []
        self.save_epi_reward = []
        self.test_reward = 0

    def get_action(self, state, epsilon, mode = "epsilon_greedy"):
        # epsilon = 0 => greedy
        # epsilon = 1 => random
        if mode == "epsilon_greedy":
            p = np.random.rand()
            if p < epsilon:
                return np.random.randint(0, self.action_kind, size = 1)[0]
                
            state_values = self.robust_q.get_v_values(state)
            max_index = np.argwhere(state_values == np.amax(state_values))
            max_index = max_index.flatten().tolist()
            if len(max_index) == 1:
                return max_index[0]
            else:
                random.shuffle(max_index)
                return max_index[0]

        elif mode == "boltzmann":
            p = np.random.rand()
            state_values = np.array(self.robust_q.get_v_values(state), dtype = np.float64)
            state_probability = np.exp(state_values / epsilon) / np.sum(np.exp(state_values / epsilon))

            for prob in state_probability:
                if math.isnan(prob):
                    print("There is nan in action value")
                    print("State Values ", state_values)
                    while(True):
                        x = 1

            sum_value = 0
            for i in range(len(state_probability)):
                sum_value = sum_value + state_probability[i]
                if p < sum_value:
                    return i
            return len(state_probability) - 1
                
        print("Wrong mode is selected")

    def q_learn(self, states, actions, q_targets, mode = "boltzmann", epsilon = TAU_END):
        for i in range(states.shape[0]):
            state = states[i]
            action = actions[i]
            q_target = q_targets[i]

            current_q_value = self.robust_q(state, action)
            update_value = (1 - self.ALPHA) * current_q_value + self.ALPHA * q_target
            self.robust_q.update_value_function(state, action, update_value, mode = mode, epsilon = epsilon)

    # 수정 필요
    def q_target(self, rewards, r_values, v_values, dones, R):
        y_k = np.asarray(v_values)
        for i in range(v_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * (R * r_values[i] + (1 - R) * v_values[i])
        return y_k

    def cal_r_value(self, states):
        r_values = []
        v_value = self.robust_q.get_v_value()
        for state in states:
            r_value = 0
            update_yet = True
            ptv = self.PTM[state]
            for next_state, probability in enumerate(ptv):
                if probability == 0:
                    continue
                if update_yet:
                    update_yet = False
                    r_value = v_value[next_state]
                else:
                    r_value = min(r_value, v_value[next_state])
            r_values.append(r_value)

        return np.array(r_values)

    # Just for test
    def cal_r_value_random(self, states):
        r_values = []
        v_value = self.robust_q.get_v_value()
        for state in states:
            r_value_list = []
            r_value = 0
            update_yet = True
            ptv = self.PTM[state]
            for next_state, probability in enumerate(ptv):
                if probability == 0:
                    continue
                r_value_list.append(v_value[next_state])
                if update_yet:
                    update_yet = False
                    r_value = v_value[next_state]
                else:
                    r_value = min(r_value, v_value[next_state])
            # print("R_Value List : ", r_value_list)
            random.shuffle(r_value_list)
            if len(r_value_list) == 0:
                r_values.append(0)
            else:
                r_values.append(r_value_list[0])

        return np.array(r_values)

    def cal_r_value_for_previous(self, states):
        r_values = []
        v_value = self.robust_q.get_v_value()
        possible_v_value = []
        for next_state, probability in enumerate(self.PTM_for_previous):
            if probability == 0:
                continue
            if v_value[next_state] == 0:
                continue
            possible_v_value.append(v_value[next_state])
        if len(possible_v_value) == 0:
            r_values.append(0)
        else:
            r_values.append(np.min(possible_v_value))

        return np.resize(r_values, len(states))

    def cal_epsilon(self, episode_ratio, mode = "exp"):
        if mode == "exp":
            a = - 1 / math.log(EPS_END / EPS_START)
            return EPS_START * math.exp(-episode_ratio / a)
        elif mode == "linear":
            return (EPS_END - EPS_START) * (episode_ratio) + EPS_START
        elif mode == "late_exp":
            if episode_ratio < 0.5:
                return 1
            else:
                episode_ratio = 2 * episode_ratio - 1
                a = - 1 / math.log(EPS_END / EPS_START)
            return EPS_START * math.exp(-episode_ratio / a)
        else:
            print("Wrong mode selected")
            return 0

    def cal_tau(self, episode_ratio, mode = "linear"):
        if mode == "linear":
            return (self.tau - TAU_START) * (episode_ratio) + TAU_START
        elif mode == "exp":
            a = - 1 / math.log(self.tau / TAU_START)
            return TAU_START * math.exp(-episode_ratio / a)
        else:
            print("Wrong mode selected")
            return 0

    def cal_v_value(self, states):
        v_table = self.robust_q.get_v_value()
        state_value = []
        for state in states:
            state_value.append(v_table[state])

        return np.array(state_value)

    def test(self, mode):
        test_reward = []
        for _ in range(int(self.TEST_STEP)):
            state, _ = self.env.reset()
            time, episode_reward, done, truncated = 0, 0, False, False
            while not done and not truncated:
                if mode == "boltzmann":
                    action = self.get_action(state, self.tau, mode = mode)
                elif mode == "epsilon_greedy":
                    action = self.get_action(state, EPS_END, mode = mode)
                next_state, reward, done, truncated, _ = self.env.step(action)

                state = next_state
                episode_reward += reward
                time += 1

            test_reward.append(episode_reward)
        self.test_reward = np.mean(test_reward)

        return self.test_reward

    def get_test_reward(self):
        return self.test_reward

    ## 에이전트 학습
    def train(self):
        r = 0
        # 에피소드마다 다음을 반복
        for ep in range(int(self.MAX_EPISODE_NUM)):

            # 에피소드 초기화
            time, episode_reward, done, truncated = 0, 0, False, False
            # 환경 초기화 및 초기 상태 관측
            state, _ = self.env.reset()
            # epsilon = self.cal_epsilon(ep / self.MAX_EPISODE_NUM, mode = "exp")
            tau = self.cal_tau(ep / self.MAX_EPISODE_NUM, mode = "linear")
            # tau = self.tau

            if self.buffer.buffer_count() > self.PTM_STEP:
                r = self.R

            while not done and not truncated:
                action = self.get_action(state, tau, mode = "boltzmann")
                # action = self.get_action(state, epsilon, mode = "epsilon_greedy")

                next_state, reward, done, truncated, _ = self.env.step(action)

                done = done or truncated
                self.PTM[state, next_state] = 1
                # self.PTM_for_previous[state] = 1

                reward = reward / self.re

                self.buffer.add_buffer(state, action, reward, next_state, done)

                # 리플레이 버퍼가 일정 부분 채워지면 학습 진행
                if self.buffer.buffer_count() > self.LEARNING_AFTER_STEP:

                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    r_values = self.cal_r_value_for_previous(states)
                    # r_values = self.cal_r_value(states)
                    v_values = self.cal_v_value(next_states)

                    y_i = self.q_target(rewards, r_values, v_values, dones, r)

                    self.q_learn(states, actions, y_i, mode = "boltzmann", epsilon = tau)

                    # 타깃 신경망 업데이트

                # 다음 스텝 준비
                state = next_state
                episode_reward += reward
                time += 1

            act = []
            q_val = self.robust_q.q_table
            for i in q_val:
                ac = i.argmax()
                act.append(ac)

            act = np.reshape(act, (4, 12))
            print("---------")
            for i in act:
                print(i)
            print("---------")
            # v_robust = self.robust_q.v_table
            # v_robust = np.reshape(v_robust, (4, 12))
            # print("---------")
            # for i in v_robust:
            #     print(i)
            # print("---------")
            # self.robust_q.print()
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_time.append(time)
            self.save_epi_reward.append(episode_reward)
    
    def get_q_table(self):
        return self.robust_q.get_q_value()

    def get_average_result(self, average_interval):
        len = int(self.MAX_EPISODE_NUM / average_interval)
        cumulative_time = np.sum(np.reshape(self.save_epi_time, (len, average_interval)), axis = 1)
        cumulative_time = np.cumsum(cumulative_time)

        average_reward = np.reshape(self.save_epi_reward, (len, average_interval))
        average_reward = np.mean(average_reward, axis = 1)

        return cumulative_time, average_reward

    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self, max_episode_num, interval):
        cumulative_time = np.sum(np.reshape(self.save_epi_time, (int(max_episode_num / interval), interval)), axis = 1)
        cumulative_time = np.cumsum(cumulative_time)

        average_reward = np.reshape(self.save_epi_reward, (int(max_episode_num / interval),interval))
        average_reward = np.mean(average_reward, axis = 1)
        print(average_reward)
        plt.subplot(2, 1, 1)
        plt.plot(self.save_epi_reward)
        plt.subplot(2, 1, 2)
        plt.plot(cumulative_time, average_reward)

        plt.show()
        return
