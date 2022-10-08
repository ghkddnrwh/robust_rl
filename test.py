import gym
from copy import deepcopy

env = gym.make("FrozenLake-v1", slippery_value = 0)

env.reset()
next_state, reward, done, truncated, _ = env.step(2)
print(next_state, reward, done, truncated)

env.set_state(6)
next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(2)
print(next_state, reward, done, truncated)

env.set_state(6)
next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(1)
print(next_state, reward, done, truncated)

next_state, reward, done, truncated, _ = env.step(2)
print(next_state, reward, done, truncated)