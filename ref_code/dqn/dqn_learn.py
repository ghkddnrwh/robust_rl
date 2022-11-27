# DQN learn (tf2 subclassing API version)
# coded by St.Watermelon

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from replaybuffer import ReplayBuffer

class Actor(Model):
    def __init__(self, action_kind):
        super(Actor, self).__init__()

        self.h1 = Dense(128, activation='relu')
        self.h2 = Dense(128, activation='relu')
        self.c = Dense(action_kind)


    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        c = self.c(x)

        return c

# Q network
class DQN(Model):

    def __init__(self, action_kind):
        super(DQN, self).__init__()

        self.h1 = Dense(128, activation='tanh')
        self.h2 = Dense(128, activation='tanh')
        self.q = Dense(action_kind)


    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        q = self.q(x)
        return q


class DQNagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 100
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.00003
        self.DQN_LEARNING_RATE = 0.0003
        self.TAU = 0.005

        self.STEPS_PER_EPOCH = 500
        self.UPDATE_AFTER = 1000
        self.UPDATE_EVERY = 50
        self.EPOCHS = 40
        # self.MAX_EP_LEN = 500


        # self.EPSILON = 1.0
        # self.EPSILON_DECAY = 0.995
        # self.EPSILON_MIN = 0.01

        self.env = env

        # get state dimension and action number
        self.state_dim = env.observation_space.shape[0]  # 4
        self.action_kind = env.action_space.n   # 2

        self.actor = Actor(self.action_kind)
        self.target_actor = Actor(self.action_kind)

        ## create Q networks
        self.dqn = DQN(self.action_kind)
        self.target_dqn = DQN(self.action_kind)

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))

        self.dqn.build(input_shape=(None, self.state_dim))
        self.target_dqn.build(input_shape=(None, self.state_dim))

        self.actor.summary()
        self.dqn.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []
        self.save_epi_test_reward = []


    def get_policy_action(self, state, actor, training = False):
        logits = actor(state, training = training)
        logp_all = tf.nn.log_softmax(logits)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis = 1)
        logp_action = tf.reduce_sum(tf.one_hot(action, depth = self.action_kind) * logp_all, axis = 1, keepdims=True)
        
        return action, logp_action


    def actor_learn(self, states, actor, critic):
        with tf.GradientTape() as tape:
            actions, log_pdfs = self.get_policy_action(states, actor, True)
            one_hot_actions = tf.one_hot(actions, self.action_kind)
            q = critic(states)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            q_values = q_values.numpy()

            loss_policy = log_pdfs * q_values
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, actor.trainable_variables))


    ## get action
    def choose_action(self, state):
        if np.random.random() <= self.EPSILON:
            return self.env.action_space.sample()
        else:
            qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
            return np.argmax(qs.numpy())


    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        phi = self.dqn.get_weights()
        target_phi = self.target_dqn.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_dqn.set_weights(target_phi)


    ## single gradient update on a single batch data
    def dqn_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            one_hot_actions = tf.one_hot(actions, self.action_kind)
            q = self.dqn(states, training=True)
            q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.dqn_opt.apply_gradients(zip(grads, self.dqn.trainable_variables))


    ## computing TD target: y_k = r_k + gamma* max Q(s_k+1, a)
    def td_target(self, rewards, target_qs, dones):
        max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(max_q.shape)
        for i in range(max_q.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * max_q[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.dqn.load_weights(path + 'cartpole_dqn.h5')

    def test(self, test_num):
        for ep in range(int(test_num)):
            time, episode_reward, done = 0, 0, False
            ep_time = 0
            state, _ = self.env.reset()

            while not done:
                action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.actor)
                action = action.numpy()[0]
                if ep_time % 100 == 0:
                    print("Action : ", action)
                    qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
                    print(qs.numpy())

                # observe reward, new_state
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                state = next_state
                episode_reward += reward
                time += 1
                ep_time += 1

            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_test_reward.append(episode_reward)
        return np.mean(self.save_epi_test_reward)

    ## train the agent
    def train(self):
        self.update_target_network(1.0)
        
        total_steps = self.STEPS_PER_EPOCH * self.EPOCHS
        time, episode_reward, done, episode_time = 0, 0, False, 0
        state, _ = self.env.reset()

        for current_step in range(int(total_steps)):
            action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.actor)
            action = action.numpy()[0]
            if current_step % 100 == 0:
                print("Action : ", action)
                qs = self.dqn(tf.convert_to_tensor([state], dtype=tf.float32))
                print(qs.numpy())

            # observe reward, new_state
            next_state, reward, done, truncated, _ = self.env.step(action)
            train_reward = reward

            # add transition to replay buffer
            self.buffer.add_buffer(state, action, train_reward, next_state, done)

            state = next_state
            episode_reward += reward
            time += 1

            if done or truncated:
                episode_time += 1
                self.save_epi_reward.append(episode_reward)
                print('Episode: ', episode_time, 'Time: ', time, "Current Step: ", current_step + 1, 'Reward: ', episode_reward)
                state, _ = self.env.reset()

                time, episode_reward = 0, 0


            if current_step >= self.UPDATE_AFTER and current_step % self.UPDATE_EVERY == 0:
                for _ in range(self.UPDATE_EVERY):
                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # predict target Q-values
                    target_qs = self.target_dqn(tf.convert_to_tensor(
                                                        next_states, dtype=tf.float32))

                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    

                    # train critic using sampled batch
                    self.dqn_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                   actions,
                                   tf.convert_to_tensor(y_i, dtype=tf.float32))

                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.actor, self.dqn)

                    # update target network
                    self.update_target_network(self.TAU)

        return self.save_epi_reward


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()