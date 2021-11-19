#coding=utf-8
import numpy as np
import time
from hri_env_v1 import HriEnv_v1
from matplotlib import pyplot
from ddpg import DDPG

MAX_EPISODES = 20
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 1000
model_path = './model'

###############################  training  ####################################
hri_env = HriEnv_v1()
s_dim = 3
a_dim = 2

ddpg_training = DDPG(a_dim, s_dim)

t1 = time.time()
reward_list = []
for i in range(MAX_EPISODES):
    s = hri_env.reset().reshape(1, s_dim)
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        # Add exploration noise
        a = ddpg_training.choose_action(s)
        a = np.array(a.reshape(a_dim, 1)).reshape(-1)
        s_, r = hri_env.step(a)
        s = s.reshape(1, s_dim)
        a = a.reshape(1, a_dim)
        s_ = s_.reshape(1, s_dim)
        R = np.array([r]).reshape(1, 1)
        ddpg_training.store_transition(s, a, R, s_)

        if ddpg_training.pointer > MEMORY_CAPACITY:
            ddpg_training.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward))
            reward_list.append(ep_reward)

print('Running time: ', time.time() - t1)

ddpg_training.save_model(model_path)
print('Model saved.')

pyplot.figure(1)
pyplot.plot(reward_list)
pyplot.ylabel('rewards')
pyplot.show()

