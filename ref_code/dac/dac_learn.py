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

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.c = Dense(action_kind)


    def call(self, x):
        x = self.h1(x)
        x = self.h2(x)
        c = self.c(x)

        return c

# Q network
class DQN(Model):

    def __init__(self):
        super(DQN, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.q = Dense(1)


    def call(self, state_action):
        x = self.h1(state_action)
        x = self.h2(x)
        q = self.q(x)

        return q


class DQNagent(object):

    def __init__(self, env):

        ## hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 100
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.DQN_LEARNING_RATE = 0.001
        self.TAU = 0.005
        # self.EPSILON = 1.0
        # self.EPSILON_DECAY = 0.995
        # self.EPSILON_MIN = 0.01

        self.env = env

        # get state dimension and action number
        self.state_dim = env.observation_space.shape[0]  # 4
        self.action_kind = env.action_space.n   # 2

        self.actor = Actor(self.action_kind)
        # self.target_actor = Actor(self.action_kind)

        ## create Q networks
        self.dqn = DQN()
        self.target_dqn = DQN()

        self.actor.build(input_shape=(None, self.state_dim))

        self.dqn.build(input_shape=(None, self.state_dim + 1))
        self.target_dqn.build(input_shape=(None, self.state_dim + 1))

        self.actor.summary()
        self.dqn.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.dqn_opt = Adam(self.DQN_LEARNING_RATE)

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []


    def get_policy_action(self, state, actor, training = False):
        logits = actor(state, training = training)
        logp_all = tf.nn.log_softmax(logits)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis = 1)
        logp_action = tf.reduce_sum(tf.one_hot(action, depth = self.action_kind) * logp_all, axis = 1, keepdims=True)
        
        return action, logp_action

    # Need to be fixed
    def actor_learn(self, states, actor, critic):
        with tf.GradientTape() as tape:
            actions, log_pdfs = self.get_policy_action(states, actor, True)
            actions = tf.reshape(actions, (self.BATCH_SIZE, 1))
            actions = tf.cast(actions, dtype = tf.float32)
            state_actions = tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype=tf.float32)

            q = critic(state_actions)
            q_values = q.numpy()

            loss_policy = log_pdfs * q_values
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, actor.trainable_variables))


    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        phi = self.dqn.get_weights()
        target_phi = self.target_dqn.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_dqn.set_weights(target_phi)


    ## single gradient update on a single batch data
    def dqn_learn(self, state_actions, td_targets, dqn):
        with tf.GradientTape() as tape:
            q_values = dqn(state_actions, training=True)
            loss = tf.reduce_mean(tf.square(q_values-td_targets))

        grads = tape.gradient(loss, dqn.trainable_variables)
        self.dqn_opt.apply_gradients(zip(grads, dqn.trainable_variables))


    ## computing TD target: y_k = r_k + gamma* max Q(s_k+1, a)
    def td_target(self, rewards, target_qs, dones):
        # max_q = np.max(target_qs, axis=1, keepdims=True)
        y_k = np.zeros(target_qs.shape)
        for i in range(target_qs.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * target_qs[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.dqn.load_weights(path + 'cartpole_dqn.h5')


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.update_target_network(1.0)

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state, _ = self.env.reset()

            while not done:
                # visualize the environment
                #self.env.render()
                # pick an action
                action, _ = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32), self.actor)
                action = action.numpy()[0]
                # observe reward, new_state
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                train_reward = reward

                # add transition to replay buffer
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:  # start train after buffer has some amounts
                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    next_actions, _ = self.get_policy_action(tf.convert_to_tensor(next_states, dtype = tf.float32), self.actor)
                    next_actions = tf.reshape(next_actions, (self.BATCH_SIZE, 1))
                    next_actions = tf.cast(next_actions, dtype = tf.float32)
                    next_state_actions = tf.convert_to_tensor(tf.concat([next_states, next_actions], axis = -1), dtype = tf.float32)

                    # predict target Q-values
                    target_qs = self.target_dqn(next_state_actions)

                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), dones)

                    actions = tf.reshape(actions, (self.BATCH_SIZE, 1))
                    actions = tf.cast(actions, dtype = tf.float32)
                    state_actions = tf.convert_to_tensor(tf.concat([states, actions], axis = -1), dtype = tf.float32)


                    # train critic using sampled batch
                    self.dqn_learn(state_actions,
                                   tf.convert_to_tensor(y_i, dtype=tf.float32), 
                                   self.dqn)

                    self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32), self.actor, self.dqn)

                    # update target network
                    self.update_target_network(self.TAU)


                # update current state
                state = next_state
                episode_reward += reward
                time += 1


            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            # self.dqn.save_weights("./save_weights/cartpole_dqn.h5")
        return self.save_epi_reward
        # np.savetxt('./save_weights/cartpole_epi_reward.txt', self.save_epi_reward)

    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()