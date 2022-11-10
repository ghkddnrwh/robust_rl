# DDPG learn (tf2 subclassing version: using chain rule to train Actor)
# coded by St.Watermelon

# 필요한 패키지 임포트
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from copy import deepcopy

from attack_replaybuffer import ReplayBuffer


## 액터 신경망
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = Dense(256, activation='relu')
        self.h2 = Dense(256, activation='relu')
        self.action = Dense(action_dim)


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        a = self.action(x)

        # 행동을 [-action_bound, action_bound] 범위로 조정
        a = tf.tanh(a)
        a *= self.action_bound
        # a = Lambda(lambda x: x*self.action_bound)(a)

        return a


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


## DDPG 에이전트
class DDPGagent(object):

    def __init__(self, env, pess_env, R = 0):

        # 하이퍼파라미터
        self.GAMMA = 0.99
        self.BATCH_SIZE = 100
        # self.BUFFER_SIZE = 1e6
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 1e-4
        self.CRITIC_LEARNING_RATE = 1e-3
        # self.STEPS_PER_EPOCH = 4000
        self.STEPS_PER_EPOCH = 200
        # self.START_STEPS = 10000
        self.START_STEPS = 200
        self.UPDATE_AFTER = 1000
        self.UPDATE_EVERY = 50
        # self.MAX_EP_LEN = 1000
        self.MAX_EP_LEN = 200
        self.TAU = 0.005
        self.EPOCHS = 400
        # self.EPOCHS = 10
        self.R = R
        self.PESS_STEP = 5000

        self.NUM_TEST_EPISODES = 100
        
        self.env = env
        self.pess_env = pess_env
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # 액터, 타깃 액터 신경망 및 크리틱, 타깃 크리틱 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)

        self.pess_actor = Actor(self.action_dim, self.action_bound)
        self.pess_target_actor = Actor(self.action_dim, self.action_bound)

        self.critic = Critic()
        self.target_critic = Critic()

        self.pess_critic = Critic()
        self.pess_target_critic = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))

        self.pess_actor.build(input_shape=(None, self.state_dim))
        self.pess_target_actor.build(input_shape=(None, self.state_dim))

        self.critic.build(input_shape = (None, self.state_dim + self.action_dim))
        self.target_critic.build(input_shape = (None, self.state_dim + self.action_dim))

        self.pess_critic.build(input_shape = (None, self.state_dim + self.action_dim))
        self.pess_target_critic.build(input_shape = (None, self.state_dim + self.action_dim))

        self.actor.summary()
        self.critic.summary()

        # 옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []
        self.save_epi_test_reward = []
        self.pess_save_epi_reward = []


    ## 신경망의 파라미터값을 타깃 신경망으로 복사
    def update_target_network(self, TAU, actor, target_actor, critic, target_critic):
        theta = actor.get_weights()
        target_theta = target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        target_actor.set_weights(target_theta)

        phi = critic.get_weights()
        target_phi = target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        target_critic.set_weights(target_phi)


    ## 크리틱 신경망 학습
    def critic_learn(self, state_actions, td_targets, critic):
        with tf.GradientTape() as tape:
            q = critic(state_actions, training=True)
            loss = tf.reduce_mean(tf.square(q-td_targets))

        grads = tape.gradient(loss, critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, critic.trainable_variables))


    ## 액터 신경망 학습
    def actor_learn(self, states, actor, critic):
        with tf.GradientTape() as tape:
            actions = actor(states, training=True)
            state_actions = tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32)
            critic_q = critic(state_actions)
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, actor.trainable_variables))


    ## Ornstein Uhlenbeck 노이즈
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)


    ## TD 타깃 계산
    def td_target(self, rewards, q_values, dones, r, r_values):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * (r * r_values[i] + (1 - r) * q_values[i])
        return y_k


    ## 신경망 파라미터 로드
    def load_weights(self, save_path):
        self.actor.load_weights(os.path.join(save_path, "robust_actor.h5"))
        self.critic.load_weights(os.path.join(save_path, "robust_crtic.h5"))

        self.pess_actor.load_weights(os.path.join(save_path, "pess_actor.h5"))
        self.pess_critic.load_weights(os.path.join(save_path, "pess_crtic.h5"))

    def save_paremeters(self, save_path):
        self.actor.save_weights(os.path.join(save_path, "robust_actor.h5"))
        self.critic.save_weights(os.path.join(save_path, "robust_crtic.h5"))

        self.pess_actor.save_weights(os.path.join(save_path, "pess_actor.h5"))
        self.pess_critic.save_weights(os.path.join(save_path, "pess_crtic.h5"))

    ## 에이전트 학습
    def train(self):
        r = 0
        self.update_target_network(1.0, self.actor, self.target_actor, self.critic, self.target_critic)
        self.update_target_network(1.0, self.pess_actor, self.pess_target_actor, self.pess_critic, self.pess_target_critic)

        total_steps = self.STEPS_PER_EPOCH * self.EPOCHS
        time, episode_reward, done, episode_time = 0, 0, False, 0
        pess_episode_reward = 0
        state, _ = self.env.reset()
        self.pess_env.reset()

        pre_noise = np.zeros(self.action_dim)
        pess_pre_noise = np.zeros(self.action_dim)

        for current_step in range(total_steps):
            exact_state = self.env.get_exact_state()
            self.pess_env.set_state(exact_state)
            # self.pess_env = deepcopy(self.env)

            action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
            action = action.numpy()[0]
            pess_action = self.pess_actor(tf.convert_to_tensor([state], dtype=tf.float32))
            pess_action = pess_action.numpy()[0]

            noise = self.ou_noise(pre_noise, dim=self.action_dim)
            pess_noise = self.ou_noise(pess_pre_noise, dim=self.action_dim)

            action = np.clip(action + noise, -self.action_bound, self.action_bound)
            pess_action = np.clip(pess_action + pess_noise, -self.action_bound, self.action_bound)

            next_state, reward, done, truncated, _ = self.env.step(action)
            pess_next_state, pess_reward, pess_done, pess_truncated, _ = self.pess_env.step(pess_action)

            done = done or truncated
            pess_done = pess_done or pess_truncated
            pess_reward = - pess_reward
            # pess_reward = pess_reward
            time += 1
            done = False if time == self.MAX_EP_LEN else done           # pendulum에서는 문제없음 다른 환경은 확인해보기 ex 특정 스텝에 도달하면 환경이 done 시그널을 True로 바꿔서 내보내는지 또 그게 환경에 어떤 영향을 미치는지
            pess_done = False if time == self.MAX_EP_LEN else pess_done           # pendulum에서는 문제없음 다른 환경은 확인해보기 ex 특정 스텝에 도달하면 환경이 done 시그널을 True로 바꿔서 내보내는지 또 그게 환경에 어떤 영향을 미치는지
            self.buffer.add_buffer(state, action, reward, next_state, done, pess_action, pess_reward, pess_next_state, pess_done)

            pre_noise = noise
            pess_pre_noise = pess_noise
            state = next_state
            episode_reward += reward
            pess_episode_reward += pess_reward

            if pess_done:
                self.pess_env.reset()

            if done or (time == self.MAX_EP_LEN):
                episode_time += 1
                self.save_epi_reward.append(episode_reward)
                self.pess_save_epi_reward.append(pess_episode_reward)
                print("Action: ", action, "Pess Action: ", pess_action)
                print("Episode Time: ", episode_time, 'Reward: ', episode_reward, "Pess Reward: ", pess_episode_reward, 'Time: ', time, 'Current Step: ', current_step + 1)
                state, _ = self.env.reset()
                self.pess_env.reset()
                pre_noise = np.zeros(self.action_dim)
                pess_pre_noise = np.zeros(self.action_dim)
                time, episode_reward = 0, 0
                pess_episode_reward = 0
                
            
            if current_step >= self.UPDATE_AFTER and current_step % self.UPDATE_EVERY == 0:
                for _ in range(self.UPDATE_EVERY):
                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states, actions, rewards, next_states, dones, pess_actions, pess_rewards, pess_next_states, pess_dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    
                    next_actions = self.target_actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    next_state_actions = tf.convert_to_tensor(tf.concat([next_states, next_actions], axis = -1), dtype = tf.float32)
                    v_target_qs = self.target_critic(next_state_actions)

                    pess_next_actions = self.pess_target_actor(tf.convert_to_tensor(pess_next_states, dtype=tf.float32))
                    pess_next_state_actions = tf.convert_to_tensor(tf.concat([pess_next_states, pess_next_actions], axis = -1), dtype = tf.float32)
                    r_target_qs = self.target_critic(pess_next_state_actions)
                    # TD 타깃 계산
                    y_i = self.td_target(rewards, v_target_qs.numpy(), dones, r, r_target_qs.numpy())
                    # 크리틱 신경망 업데이트
                    state_actions = tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32)
                    self.critic_learn(state_actions,
                                    tf.convert_to_tensor(y_i, dtype=tf.float32), 
                                    self.critic)

                    pess_target_qs = self.pess_target_critic(pess_next_state_actions)            
                    pess_y_i = self.td_target(pess_rewards, pess_target_qs.numpy(), pess_dones, 0, pess_target_qs.numpy())
                    pess_state_actions = tf.convert_to_tensor(tf.concat([states, pess_actions], axis = -1), dtype=tf.float32)
                    self.critic_learn(pess_state_actions,
                                    tf.convert_to_tensor(pess_y_i, dtype=tf.float32), 
                                    self.pess_critic)
                    # 액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.actor, self.critic)
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.pess_actor, self.pess_critic)
                    # 타깃 신경망 업데이트
                    self.update_target_network(self.TAU, self.actor, self.target_actor, self.critic, self.target_critic)
                    self.update_target_network(self.TAU, self.pess_actor, self.pess_target_actor, self.pess_critic, self.pess_target_critic)

            if current_step == self.PESS_STEP:
                r = self.R
        
        return self.save_epi_reward
                
    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
