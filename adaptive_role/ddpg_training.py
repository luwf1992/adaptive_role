#coding=utf-8
import numpy as np
import time
from hri_env_v2 import HriEnv_v2
from matplotlib import pyplot
from ddpg import DDPG

MAX_EPISODES = 100
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000
model_path = './model'

###############################  training  ####################################
hri_env = HriEnv_v2()
s_dim = 6
a_dim = hri_env.task_num
min_action = [0.] * a_dim
max_action = [1.] * a_dim

ddpg_training = DDPG(a_dim, s_dim)

t1 = time.time()
reward_list = []
sigma = 1.
for i in range(MAX_EPISODES):
    s = hri_env.reset()
    ep_reward = 0.
    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        a = ddpg_training.choose_action(s, sigma)
        a = a.numpy().reshape(-1)
        # noise = np.random.normal(0., sigma)
        # a[0] += noise
        # a[1] -= noise / 2
        # a[2] -= noise / 2
        # a = np.clip(a, min_action, max_action)
        # print(a)
        s_, r = hri_env.step(a)
        ddpg_training.store_transition(s, a, r, s_)

        if ddpg_training.pointer > MEMORY_CAPACITY:
            sigma *= .995
            ddpg_training.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %f' % ep_reward, 'Explore: %.2f' % sigma,
                  ' task index', hri_env.task_index)
            reward_list.append(ep_reward)

print('Running time: ', time.time() - t1)

ddpg_training.save_model(model_path)
print('Model saved.')

pyplot.figure(1)
pyplot.plot(reward_list)
pyplot.ylabel('rewards')
pyplot.show()

