# Replay Buffer
# coded by St.Watermelon

import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## save to buffer
    def add_buffer(self, state, action, reward, next_state, done, pess_action, pess_reward, pess_next_state, pess_done):
        transition = (state, action, reward, next_state, done, pess_action, pess_reward, pess_next_state, pess_done)

        # check if buffer is full
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    ## sample a batch
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # return a batch of transitions
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        pess_actions = np.asarray([i[5] for i in batch])
        pess_rewards = np.asarray([i[6] for i in batch])
        pess_next_states = np.asarray([i[7] for i in batch])
        pess_dones = np.asarray([i[8] for i in batch])
        
        return states, actions, rewards, next_states, dones, pess_actions, pess_rewards, pess_next_states, pess_dones


    ## Current buffer occupation
    def buffer_count(self):
        return self.count


    ## Clear buffer
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0