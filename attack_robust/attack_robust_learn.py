# SAC learn: Two Q nets (tf2 subclassing version)
# coded by St.Watermelon

# 필요한 패키지 임포트
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

import math
import random

from replaybuffer import ReplayBuffer

class VNetwork(Model):
    def __init__(self):
        super(VNetwork, self).__init__()

        self.x1 = Dense(100, activation='relu')
        self.h1 = Dense(100, activation='relu')
        self.v = Dense(1, activation='linear')

    def call(self, state):
        h = self.x1(state)
        h = self.h1(h)
        v = self.v(h)
        return v

## 크리틱 신경망
class QNetwork(Model):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.x1 = Dense(50, activation='relu')
        self.a1 = Dense(50, activation='relu')
        self.h2 = Dense(100, activation='relu')
        self.q = Dense(1, activation='linear')


    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        a = self.a1(action)
        h = concatenate([x, a], axis=-1)
        h = self.h2(h)
        q = self.q(h)
        return q

## SAC 에이전트
class SACagent(object):
    def __init__(self, env):
        self.GAMMA = 0.95
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 20000
        self.CRITIC_LEARNING_RATE = 0.001

        # 환경
        self.env = env
        self.state_kind = env.observation_space.n
        self.action_kind = env.action_space.n

        self.q_network = QNetwork()
        self.target_q_network = QNetwork()

        self.v_network = VNetwork()
        self.target_v_network = VNetwork()

        state_in = Input((1))
        action_in = Input((1))
        self.q_network([state_in, action_in])
        self.target_q_network([state_in, action_in])

        self.v_network([state_in])
        self.target_v_network([state_in])

        self.q_network.summary()
        self.v_network.summary()

        # 옵티마이저
        self.q_network_opt = Adam(self.CRITIC_LEARNING_RATE)
        self.v_network_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []

    def get_v_values(self, state):
        v_values = []
        for action in self.action_kind:
            v_values.append(self.q_network([state, action]))
        return np.array(v_values)

    ## 행동 샘플링
    def get_action(self, state, epsilon, mode = "epsilon_greedy"):
        if mode == "epsilon_greedy":
            p = np.random.rand()
            if p < epsilon:
                return np.random.randint(0, self.action_kind, size = 1)[0]
                
            state_values = self.get_v_values(state)
            max_index = np.argwhere(state_values == np.amax(state_values))
            max_index = max_index.flatten().tolist()
            if len(max_index) == 1:
                return max_index[0]
            else:
                random.shuffle(max_index)
                return max_index[0]

        elif mode == "boltzmann":
            p = np.random.rand()
            state_values = np.array(self.get_v_values(state), dtype = np.float64)
            state_values = np.exp(state_values / epsilon) / np.sum(np.exp(state_values / epsilon))

            for state in state_values:
                if math.isnan(state):
                    print("There is nan in action value")
                    while(True):
                        x = 1

            sum_value = 0
            for i in range(len(state_values)):
                sum_value = sum_value + state_values[i]
                if p < sum_value:
                    return i
            return len(state_values) - 1
                
        print("Wrong mode is selected")



    ## 신경망의 파라미터값을 타깃 신경망으로 복사
    def update_target_network(self, TAU):
        phi_1 = self.critic_1.get_weights()
        phi_2 = self.critic_2.get_weights()
        target_phi_1 = self.target_critic_1.get_weights()
        target_phi_2 = self.target_critic_2.get_weights()
        for i in range(len(phi_1)):
            target_phi_1[i] = TAU * phi_1[i] + (1 - TAU) * target_phi_1[i]
            target_phi_2[i] = TAU * phi_2[i] + (1 - TAU) * target_phi_2[i]
        self.target_critic_1.set_weights(target_phi_1)
        self.target_critic_2.set_weights(target_phi_2)


    ## Q1, Q2 신경망 학습
    def q_learn(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q_1 = self.q_network([states, actions], training=True)
            loss_1 = tf.reduce_mean(tf.square(q_1-q_targets))

        grads_1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            q_2 = self.critic_2([states, actions], training=True)
            loss_2 = tf.reduce_mean(tf.square(q_2-q_targets))

        grads_2 = tape.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_2_opt.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))


    ## 시간차 타깃 계산
    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor_2q.h5')
        self.critic_1.load_weights(path + 'pendulum_critic_12q.h5')
        self.critic_2.load_weights(path + 'pendulum_critic_22q.h5')


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 타깃 신경망 초기화
        self.update_target_network(1.0)

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()

            while not done:
                # 환경 가시화
                #self.env.render()
                # 행동 샘플링
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # 학습용 보상 설정
                train_reward = (reward + 8) / 8
                # 리플레이 버퍼에 저장
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                # 리플레이 버퍼가 일정 부분 채워지면 학습 진행
                if self.buffer.buffer_count() > 1000:

                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # Q 타깃 계산
                    next_mu, next_std = self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

                    target_qs_1 = self.target_critic_1([next_states, next_actions])
                    target_qs_2 = self.target_critic_2([next_states, next_actions])
                    target_qs = tf.math.minimum(target_qs_1, target_qs_2)

                    target_qi = target_qs - self.ALPHA * next_log_pdf

                    # TD 타깃 계산
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    # Q1, Q2 신경망 업데이트
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # 액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))

                    # 타깃 신경망 업데이트
                    self.update_target_network(self.TAU)

                # 다음 스텝 준비
                state = next_state
                episode_reward += reward
                time += 1

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)


            # 에피소드마다 신경망 파라미터를 파일에 저장
            self.actor.save_weights("./save_weights/pendulum_actor_2q.h5")
            self.critic_1.save_weights("./save_weights/pendulum_critic_12q.h5")
            self.critic_2.save_weights("./save_weights/pendulum_critic_22q.h5")

        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./save_weights/pendulum_epi_reward_2q.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
