# SAC learn: Two Q nets (tf2 subclassing version)
# coded by St.Watermelon

# 필요한 패키지 임포트
from threading import currentThread
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp

from replaybuffer import ReplayBuffer

EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

## 액터 신경망
class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound 

        self.h1 = Dense(256, activation='relu')
        self.h2 = Dense(256, activation='relu')
        self.mu = Dense(action_dim)
        self.log_std = Dense(action_dim)


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        return mu, std

    ## 행동을 샘플링하고 log-pdf 계산
    def sample_normal(self, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        
        log_pdf = normal_prob.log_prob(action)
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)
        log_pdf -= tf.reduce_sum(2*(np.log(2) - action - tf.nn.softplus(-2*action)), axis=1, keepdims=True)

        mu = tf.tanh(mu)
        mu *= self.action_bound     # 이 두개의 위치에 따른 성능 차이 확인해 보기
        action = tf.tanh(action)
        action *= self.action_bound

        return mu, action, log_pdf


## 크리틱 신경망
class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(256, activation='relu')
        self.h2 = Dense(256, activation='relu')
        self.q = Dense(1)


    def call(self, state_action):
        x = self.h1(state_action)
        x = self.h2(x)
        q = self.q(x)
        return q


## SAC 에이전트
class SACagent(object):

    def __init__(self, env):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 100
        # self.BUFFER_SIZE = 1e6
        self.BUFFER_SIZE = 20000
        self.LEARNING_RATE = 1e-3
        self.ALPHA = 0.2
        # self.STEPS_PER_EPOCH = 4000
        self.STEPS_PER_EPOCH = 200
        # self.START_STEPS = 10000
        self.START_STEPS = 200
        self.UPDATE_AFTER = 1000
        self.UPDATE_EVERY = 50
        # self.MAX_EP_LEN = 1000
        self.MAX_EP_LEN = 200
        self.TAU = 0.005
        # self.EPOCHS = 100
        self.EPOCHS = 100

        self.NUM_TEST_EPISODES = 10
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # 액터 신경망 및 Q1, Q2 타깃 Q1, Q2 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)
        self.actor.build(input_shape=(None, self.state_dim))

        self.critic_1 = Critic()
        self.target_critic_1 = Critic()

        self.critic_2 = Critic()
        self.target_critic_2 = Critic()

        self.critic_1.build(input_shape = (None, self.state_dim + self.action_dim))
        self.target_critic_1.build(input_shape = (None, self.state_dim + self.action_dim))
        self.critic_2.build(input_shape = (None, self.state_dim + self.action_dim))
        self.target_critic_2.build(input_shape = (None, self.state_dim + self.action_dim))

        self.actor.summary()
        self.critic_1.summary()
        self.critic_2.summary()

        # 옵티마이저
        self.actor_opt = Adam(self.LEARNING_RATE)
        self.critic_1_opt = Adam(self.LEARNING_RATE)
        self.critic_2_opt = Adam(self.LEARNING_RATE)

        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []


    ## 행동 샘플링
    def get_action(self, state, deterministic = False):
        mu, std = self.actor(state)
        mu, action, _ = self.actor.sample_normal(mu, std)

        if deterministic:
            return mu.numpy()[0]
        return action.numpy()[0]


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
    def critic_learn(self, state_actions, q_targets):
        with tf.GradientTape() as tape:
            q_1 = self.critic_1(state_actions, training=True)
            loss_1 = tf.reduce_mean(0.5 * tf.square(q_1-q_targets))

        grads_1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            q_2 = self.critic_2(state_actions, training=True)
            loss_2 = tf.reduce_mean(0.5 * tf.square(q_2-q_targets))

        grads_2 = tape.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_2_opt.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))


    ## 액터 신경망 학습
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states, training=True)
            _, actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            # soft_q_1 = self.critic_1(tf.concat([states, actions], axis = -1))
            # soft_q_2 = self.critic_2(tf.concat([states, actions], axis = -1))
            soft_q_1 = self.critic_1(tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32))
            soft_q_2 = self.critic_2(tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32))
            soft_q = tf.math.minimum(soft_q_1, soft_q_2)

            loss = tf.reduce_mean(self.ALPHA * log_pdfs - soft_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## 시간차 타깃 계산
    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k

    ## 에이전트 학습
    def train(self):
        self.update_target_network(1.0)
        total_steps = self.STEPS_PER_EPOCH * self.EPOCHS
        time, episode_reward, done, episode_time = 0, 0, False, 0
        state, _ = self.env.reset()

        for current_step in range(total_steps):
            if current_step > self.START_STEPS:
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            else:
                action = self.env.action_space.sample()

            next_state, reward, done, truncated, _ = self.env.step(action) #Terminate 포함인지 확인
            done = done or truncated
            time += 1
            done = False if time == self.MAX_EP_LEN else done
            self.buffer.add_buffer(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done or (time == self.MAX_EP_LEN):
                episode_time += 1
                self.save_epi_reward.append(episode_reward)
                print("Episode Time: ", episode_time, 'Reward: ', episode_reward, 'Time: ', time, 'Current Step: ', current_step + 1)
                state, _ = self.env.reset()
                time, episode_reward = 0, 0
                
            if current_step >= self.UPDATE_AFTER and current_step % self.UPDATE_EVERY == 0:
                for _ in range(self.UPDATE_EVERY):
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    next_mu, next_std = self.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    _, next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

                    target_qs_1 = self.target_critic_1(tf.convert_to_tensor(tf.concat([next_states, next_actions], axis = -1), dtype = tf.float32))
                    target_qs_2 = self.target_critic_2(tf.convert_to_tensor(tf.concat([next_states, next_actions], axis = -1), dtype = tf.float32))
                    target_qs = tf.math.minimum(target_qs_1, target_qs_2)

                    target_qi = target_qs - self.ALPHA * next_log_pdf
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    self.critic_learn(tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32),
                                    tf.convert_to_tensor(y_i, dtype=tf.float32))

                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                    self.update_target_network(self.TAU)
        
        return self.save_epi_reward


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
