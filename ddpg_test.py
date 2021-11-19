#coding=utf-8
import numpy as np
from hri_env_v1 import HriEnv_v1
import matplotlib.pyplot as plt
from ddpg import DDPG

s_dim = 3
a_dim = 2
model_path = './model'
hri_env = HriEnv_v1()
MAX_EP_STEPS = 200

ddpg_test = DDPG(a_dim, s_dim)
ddpg_test.load_weights(model_path)
ddpg_test.eval()
policy = lambda x: ddpg_test.choose_action(x)
# print(policy([0, 0, 0]))

s = hri_env.reset()
print('task_index')
print(hri_env.task_index)
ep_reward = 0

b1_list = []
b2_list = []

# f1 = open('data/admittance_control/f_h.csv', mode='w')
# f1_writer = csv.writer(f1, delimiter=',')

for i in range(MAX_EP_STEPS):
    # a = policy(s.reshape(1, 9))
    a = ddpg_test.choose_action(s.reshape(1, s_dim))
    a = np.array(a).reshape(a_dim, 1)
    s_, r = hri_env.step(a)
    s = s_
    ep_reward += r
    print(a)
    b1_list.append(a[0])
    b2_list.append(a[1])

    # if i == MAX_EP_STEPS-1:
    #     print('total reward: {:f}'.format(ep_reward))

plt.figure(1)
plt.plot(b1_list, label='task 1')
plt.plot(b2_list, label='task 2')
plt.legend()

plt.show()

