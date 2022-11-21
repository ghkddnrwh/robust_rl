# PPO learn (tf2 subclassing API version)
# coded by St.Watermelon

# 필요한 패키지 임포트
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt


## PPO 액터 신경망
class Actor(Model):

    def __init__(self, action_kind):
        super(Actor, self).__init__()
        self.action_kind = action_kind

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.c = Dense(action_kind)


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        c = self.c(x)

        return c


## PPO 크리틱 신경망
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.v = Dense(1)

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        v = self.v(x)
        return v


## PPO 에이전트 클래스
class PPOagent(object):

    def __init__(self, env):

        # 하이퍼파라미터
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.97
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0003
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 5

        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        # 행동 차원
        # self.action_dim = env.action_space.shape[0]
        self.action_kind = env.action_space.n
        # 행동의 최대 크기
        # self.action_bound = env.action_space.high[0]
        # 표준편차의 최솟값과 최댓값 설정
        # self.std_bound = [1e-2, 1.0]

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.action_kind)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_dim))
        self.critic.build(input_shape=(None, self.state_dim))

        self.actor.summary()
        self.critic.summary()

        # 옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []


    ## 로그-정책 확률밀도함수 계산
    # def log_pdf(self, mu, std, action):
    #     std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
    #     var = std ** 2
    #     log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
    #     return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)


    ## 액터 신경망으로 정책의 평균, 표준편차를 계산하고 행동 샘플링
    def get_policy_action(self, state):
        # print("Get Policy Action")
        logits = self.actor(state)
        # print(logits.numpy())
        logp_all = tf.nn.log_softmax(logits)
        # print(logp_all.numpy())
        # print(np.exp(logp_all.numpy()))
        # tf.random.categorical(tf.math.log([[0.5, 0.5]]), 5)
        # action = tf.argmax(logits, axis = 1)
        # print(action1)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis = 1)
        # print(action)
        # print(action.numpy())
        # action = tf.squeeze(tf.multinomial(logits,1), axis=1)
        # print(tf.one_hot(action, depth = self.action_kind).numpy())
        logp_action = tf.reduce_sum(tf.one_hot(action, depth = self.action_kind) * logp_all, axis=1)
        # print(logp_action.numpy())
        # print("Get Policy Action Out")
        return action, logp_action


    ## GAE와 시간차 타깃 계산
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets


    ## 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    ## 액터 신경망 학습
    def actor_learn(self, log_old_policy_pdf, states, actions, gaes):
        # print("Actor learn IN")

        with tf.GradientTape() as tape:
            # 현재 정책 확률밀도함수
            # print(actions)
            # mu_a, std_a = self.actor(states, training=True)
            logits = self.actor(states)
            # print(logits.numpy())
            logp_all = tf.nn.log_softmax(logits)
            # print(logp_all.numpy())
            # action = tf.squeeze(tf.multinomial(logits,1), axis=1)
            # print(tf.one_hot(actions, depth=self.action_kind).numpy())
            log_policy_pdf = tf.reduce_sum(tf.one_hot(actions, depth=self.action_kind) * logp_all, axis=1)
            # print(log_policy_pdf.numpy())

            # 현재와 이전 정책 비율
            ratio = tf.exp(log_policy_pdf - log_old_policy_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            loss = tf.reduce_mean(surrogate)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        # print("Actor learn OUT\n")

    ## 크리틱 신경망 학습
    def critic_learn(self, states, td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.critic(states, training=True)
            loss = tf.reduce_mean(tf.square(td_hat-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 배치 초기화
        batch_state, batch_action, batch_reward = [], [], []
        batch_log_old_policy_pdf = []

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state, _ = self.env.reset()

            while not done:

                # 환경 가시화
                #self.env.render()
                # 이전 정책의 평균, 표준편차를 계산하고 행동 샘플링
                # mu_old, std_old, action = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
                action, log_old_policy_pdf = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
                # 행동 범위 클리핑
                # action = np.clip(action, -self.action_bound, self.action_bound)
                # 이전 정책의 로그 확률밀도함수 계산
                # var_old = std_old ** 2
                # log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                # log_old_policy_pdf = np.sum(log_old_policy_pdf)
                # 다음 상태, 보상 관측
                action = action.numpy()[0]
                # print("Action : ", action)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                # shape 변환
                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1])
                reward = np.reshape(reward, [1, 1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])
                # 학습용 보상 설정
                # train_reward = (reward + 8) / 8
                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)

                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state
                    episode_reward += reward[0, 0]
                    time += 1
                    continue

                # 배치가 채워지면, 학습 진행
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                # 배치 비움
                batch_state, batch_action, batch_reward, = [], [], []
                batch_log_old_policy_pdf = []
                # GAE와 시간차 타깃 계산
                next_v_value = self.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32))
                gaes, y_i = self.gae_target(rewards, v_values.numpy(), next_v_value.numpy(), done)
                # print(y_i.shape)

                # 에포크만큼 반복
                for _ in range(self.EPOCHS):
                    # 액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(log_old_policy_pdfs, dtype=tf.float32),
                                     tf.convert_to_tensor(states, dtype=tf.float32),
                                     tf.convert_to_tensor(actions, dtype=tf.int32),
                                     tf.convert_to_tensor(gaes, dtype=tf.float32))
                    # 크리틱 신경망 업데이트
                    self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                # 다음 에피소드를 위한 준비
                state = next_state
                episode_reward += reward[0, 0]
                time += 1

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)

        #     # 에피소드 10번마다 신경망 파라미터를 파일에 저장
        #     if ep % 10 == 0:
        #         self.actor.save_weights("./save_weights/pendulum_actor.h5")
        #         self.critic.save_weights("./save_weights/pendulum_critic.h5")

        # # 학습이 끝난 후, 누적 보상값 저장
        # np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

