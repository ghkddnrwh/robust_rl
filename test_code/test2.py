import gym

import collections

import random

import numpy as np

import matplotlib.pyplot as plt


import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim


#Hyperparameters

learning_rate = 0.001

gamma = 0.980

buffer_limit = 100000

batch_size = 2000


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, data):
        self.buffer.append(data)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):

        return len(self.buffer)


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(2, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()

        if coin < epsilon:
            return random.randint(0,2)
        else : 
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(15):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('MountainCar-v0')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0 
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    render = False
    max_position = -0.4
    success = 0
    count = 0
    positions = np.ndarray([0,2])

    for n_epi in range(800):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/100))
        s, _ = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon) 
            s_prime, r, done, truncated, info = env.step(a)
            done = done or truncated
            done_mask = 0.0 if done else 1.0

            if a != 3:
                count += 1

            if s_prime[1]>0: #오른쪽으로 가는 속도에 더 큰 리워드를 받도록 한다.
                r=((s_prime[0]+0.5)*10)**2/10+15*s_prime[1]-count/400 #위치에 따라 리워드를 이차함수 형태로 가중치를 받는다.
            else:
                r = ((s_prime[0]+0.5)*10)**2/10-count/400

            if s_prime[0] > max_position:
                max_position = s_prime[0]
                positions = np.append(positions, [[n_epi, max_position]], axis=0)

            if s_prime[0] >= 0.5: #flag 위치가 0.5
                success += 1 #flag에 닿으면 성공
                r = 20 #성공하면 리워드 20을 받는다.
            else:
                score -= 1

            memory.put((s,a,r,s_prime, done_mask))
            s = s_prime

            if done:
                count = 0
                max_position = -0.4
                break

            if memory.size()>8000:
                train(q, q_target, memory, optimizer)

            if n_epi%print_interval==0 and n_epi!=0:
                q_target.load_state_dict(q.state_dict())

        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, success rate : {}%".format(n_epi, score/print_interval, memory.size(), epsilon*100, success/print_interval*100))
        score = 0.0
        success = 0

        env.close()


if __name__ == '__main__':
    main()