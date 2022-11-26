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

## PPO 액터 신경망
class Actor(Model):

    def __init__(self, action_kind):
        super(Actor, self).__init__()
        self.action_kind = action_kind

        self.h1 = Dense(128, activation='relu')
        self.h2 = Dense(128, activation='relu')
        self.c = Dense(action_kind)


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        c = self.c(x)

        return c

## 크리틱 신경망
class Critic(Model):

    def __init__(self, action_kind):
        super(Critic, self).__init__()

        self.action_kind = action_kind

        self.h1 = Dense(128, activation='tanh')
        self.h2 = Dense(128, activation='tanh')
        self.q = Dense(self.action_kind)


    def call(self, state_action):
        x = self.h1(state_action)
        x = self.h2(x)
        q = self.q(x)
        return q


## DDPG 에이전트
class DQNAgent(object):
    def __init__(self, env, R = 0):

        # 하이퍼파라미터
        self.GAMMA = 0.99
        self.BATCH_SIZE = 100
        # self.BUFFER_SIZE = 1e6
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 1e-4
        self.CRITIC_LEARNING_RATE = 1e-3
        # self.CRITIC_LEARNING_RATE = R
        # self.STEPS_PER_EPOCH = 4000
        self.STEPS_PER_EPOCH = 500
        # self.START_STEPS = 10000
        self.START_STEPS = 200
        self.UPDATE_AFTER = 1000
        self.UPDATE_EVERY = 50
        # self.MAX_EP_LEN = 1000
        self.MAX_EP_LEN = 500
        self.TAU = 0.005
        # self.EPOCHS = 150
        self.EPOCHS = 120
        # self.EPOCHS = 40
        self.R = R
        self.PESS_STEP = 5000

        self.EPSILON = 1.0
        self.EPSILON_DECAY = 0.9995
        self.EPSILON_MIN = 0.01

        self.NUM_TEST_EPISODES = 10
        
        self.env = env
        self.pess_env = deepcopy(self.env)
        
        self.state_dim = env.observation_space.shape[0]
        self.action_kind = env.action_space.n

        self.actor = Actor(self.action_kind)
        self.target_actor = Actor(self.action_kind)

        self.pess_actor = Actor(self.action_kind)
        self.pess_target_actor = Actor(self.action_kind)

        self.critic = Critic(self.action_kind)
        self.target_critic = Critic(self.action_kind)

        self.pess_critic = Critic(self.action_kind)
        self.pess_target_critic = Critic(self.action_kind)

        self.actor.build(input_shape = (None, self.state_dim))
        self.target_actor.build(input_shape = (None, self.state_dim))

        self.pess_actor.build(input_shape = (None, self.state_dim))
        self.pess_target_actor.build(input_shape = (None, self.state_dim))

        self.critic.build(input_shape = (None, self.state_dim))
        self.target_critic.build(input_shape = (None, self.state_dim))

        self.pess_critic.build(input_shape = (None, self.state_dim))
        self.pess_target_critic.build(input_shape = (None, self.state_dim))

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
        self.pess_save_epi_test_reward = []


    ## 액터 신경망으로 정책의 평균, 표준편차를 계산하고 행동 샘플링
    def get_policy_action(self, state, actor, training = False):
        logits = actor(state, training = training)
        logp_all = tf.nn.log_softmax(logits)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis = 1)
        logp_action = tf.reduce_sum(tf.one_hot(action, depth = self.action_kind) * logp_all, axis = 1, keepdims=True)
        
        return action, logp_action

    ## get action
    # def choose_action(self, state, critic, epsilon):
    #     if np.random.random() <= epsilon:
    #         return self.env.action_space.sample()
    #     else:
    #         qs = critic(tf.convert_to_tensor([state], dtype=tf.float32))
    #         return np.argmax(qs.numpy())

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

    def actor_learn(self, states, actor, critic):
        with tf.GradientTape() as tape:
            actions, log_pdfs = self.get_policy_action(states, actor, True)
            one_hot_actions = tf.one_hot(actions, self.action_kind)
            q = critic(states)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            q_values = q_values.numpy()
            # actions, _ = self.get_policy_action(states, actor, training = True)
            # one_hot_actions = tf.one_hot(actions, self.action_kind)
            # q = critic(states)
            # q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)

            loss_policy = log_pdfs * q_values
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, actor.trainable_variables))

    def critic_learn(self, states, actions, td_targets, critic):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_kind)
            q = critic(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, critic.trainable_variables))

    def td_target(self, rewards, v_target_qs, dones, r, r_target_qs):
        y_k = np.zeros(v_target_qs.shape)
        for i in range(v_target_qs.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * (r * r_target_qs[i] + (1 - r) * v_target_qs[i])
        return y_k

    ## 신경망 파라미터 로드
    def load_weights(self, save_path):
        self.actor.load_weights(os.path.join(save_path, "robust_actor.h5"))
        self.pess_actor.load_weights(os.path.join(save_path, "pess_actor.h5"))

        self.critic.load_weights(os.path.join(save_path, "robust_critic.h5"))
        self.pess_critic.load_weights(os.path.join(save_path, "pess_critic.h5"))

    def save_paremeters(self, save_path):
        self.actor.save_weights(os.path.join(save_path, "robust_actor.h5"))
        self.pess_actor.save_weights(os.path.join(save_path, "pess_actor.h5"))

        self.critic.save_weights(os.path.join(save_path, "robust_critic.h5"))
        self.pess_critic.save_weights(os.path.join(save_path, "pess_critic.h5"))

    def test(self, perturb = 0):
        same_count = 0
        diff_count = 0

        self.save_epi_test_reward = []
        for ep in range(int(self.NUM_TEST_EPISODES)):
            time, episode_reward, done = 0, 0, False
            state, _ = self.env.reset()

            while not done:
                # action = self.choose_action(state, self.critic, 0)
                # pess_action = self.choose_action(state, self.pess_critic, 0)
                p = np.random.rand()
                if p < perturb:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.actor)
                    action = action.numpy()[0]
                
                    pess_action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.pess_actor)
                    pess_action = pess_action.numpy()[0]

                    if pess_action == action:
                        same_count += 1
                    else:
                        diff_count += 1
                # exact_state = self.env.get_exact_state()
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                state = next_state
                episode_reward += reward
                time += 1

            self.save_epi_test_reward.append(episode_reward)
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
        print("Same count : ", same_count)
        print("Diff count : ", diff_count)
        return np.mean(self.save_epi_test_reward)


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

        same_count = 0
        diff_count = 0

        ep_same_count = 0
        ep_diff_count = 0

        for current_step in range(total_steps):
            # exact_state = self.env.get_exact_state()
            # self.pess_env.set_state(exact_state)
            self.pess_env = deepcopy(self.env)

            # action = self.choose_action(state, self.critic, self.EPSILON)
            action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.actor)
            action = action.numpy()[0]
            # pess_action = self.choose_action(state, self.pess_critic, self.EPSILON)
            pess_action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.pess_actor)
            pess_action = pess_action.numpy()[0]
            if current_step % 100 == 0:
                print("Action : ", action, pess_action)
                act = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                pess_act = self.pess_actor(tf.convert_to_tensor([state], dtype=tf.float32))
                print(act.numpy(), pess_act.numpy())
                clt = self.critic(tf.convert_to_tensor([state], dtype = tf.float32))
                pess_clt = self.pess_critic(tf.convert_to_tensor([state], dtype = tf.float32))
                print(clt.numpy(), pess_clt.numpy())
            # print(action)

            if current_step >= self.PESS_STEP:
                if pess_action == action:
                    same_count += 1
                    ep_same_count += 1
                else:
                    diff_count += 1
                    ep_diff_count += 1

            next_state, reward, done, truncated, _ = self.env.step(action)
            pess_next_state, pess_reward, pess_done, pess_truncated, _ = self.pess_env.step(pess_action)

            done = done or truncated
            # pess_done = pess_done or pess_truncated
            pess_reward = - pess_reward
            time += 1
            done = False if time == self.MAX_EP_LEN else done           # pendulum에서는 문제없음 다른 환경은 확인해보기 ex 특정 스텝에 도달하면 환경이 done 시그널을 True로 바꿔서 내보내는지 또 그게 환경에 어떤 영향을 미치는지
            # pess_done = False if time == self.MAX_EP_LEN else pess_done           # pendulum에서는 문제없음 다른 환경은 확인해보기 ex 특정 스텝에 도달하면 환경이 done 시그널을 True로 바꿔서 내보내는지 또 그게 환경에 어떤 영향을 미치는지
            self.buffer.add_buffer(state, action, reward, next_state, done, pess_action, pess_reward, pess_next_state, pess_done)

            state = next_state
            episode_reward += reward
            pess_episode_reward += pess_reward

            if pess_done or pess_truncated:
                self.pess_env.reset()

            if current_step >= self.START_STEPS and self.EPSILON > self.EPSILON_MIN:
                self.EPSILON *= self.EPSILON_DECAY    

            if done or (time == self.MAX_EP_LEN):
                episode_time += 1
                self.save_epi_reward.append(episode_reward)
                self.pess_save_epi_reward.append(pess_episode_reward)
                # print("Action: ", action, "Pess Action: ", pess_action)
                print("Same Count : ", ep_same_count, "Diff Count : ", ep_diff_count)
                print("Episode Time: ", episode_time, 'Reward: ', episode_reward, "Pess Reward: ", pess_episode_reward, 'Time: ', time, 'Current Step: ', current_step + 1)
                state, _ = self.env.reset()
                self.pess_env.reset()
                time, episode_reward = 0, 0
                pess_episode_reward = 0
                ep_same_count = ep_diff_count = 0

            if current_step >= self.UPDATE_AFTER and current_step % self.UPDATE_EVERY == 0:
                for _ in range(self.UPDATE_EVERY):
                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states, actions, rewards, next_states, dones, pess_actions, pess_rewards, pess_next_states, pess_dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    v_next_actions, _ = self.get_policy_action(next_states, self.target_actor)
                    r_next_actions, _ = self.get_policy_action(pess_next_states, self.target_actor)
                    pess_next_actions, _ = self.get_policy_action(pess_next_states, self.pess_target_actor)

                    v_one_hot_actions = tf.one_hot(v_next_actions, self.action_kind)
                    r_one_hot_actions = tf.one_hot(r_next_actions, self.action_kind)
                    pess_one_hot_actions = tf.one_hot(pess_next_actions, self.action_kind)
                    
                    v_target_qs = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    r_target_qs = self.critic(tf.convert_to_tensor(pess_next_states, dtype = tf.float32))
                    pess_target_qs = self.pess_critic(tf.convert_to_tensor(pess_next_states, dtype=tf.float32))

                    v_target_qs = tf.reduce_sum(v_one_hot_actions * v_target_qs, axis=1, keepdims=True)
                    r_target_qs = tf.reduce_sum(r_one_hot_actions * r_target_qs, axis=1, keepdims=True)
                    pess_target_qs = tf.reduce_sum(pess_one_hot_actions * pess_target_qs, axis=1, keepdims=True)
                    # print(v_target_qs.numpy().shape)
                
                    y_i = self.td_target(rewards, v_target_qs.numpy(), dones, r, r_target_qs.numpy())
                    pess_y_i = self.td_target(pess_rewards, pess_target_qs.numpy(), pess_dones, 0, pess_target_qs.numpy())

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                   actions,
                                   tf.convert_to_tensor(y_i, dtype=tf.float32), self.critic)

                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                   pess_actions,
                                   tf.convert_to_tensor(pess_y_i, dtype=tf.float32), self.pess_critic)

                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.actor, self.critic)
                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.pess_actor, self.pess_critic)
                    
                    self.update_target_network(self.TAU, self.actor, self.target_actor, self.critic, self.target_critic)
                    self.update_target_network(self.TAU, self.pess_actor, self.pess_target_actor, self.pess_critic, self.pess_target_critic)

            if current_step == self.PESS_STEP:
                r = self.R
        
        print("Same count : ", same_count)
        print("Diff count : ", diff_count)
        return self.save_epi_reward
                
    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
