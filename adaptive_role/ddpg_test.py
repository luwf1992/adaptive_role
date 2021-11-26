#coding=utf-8
import numpy as np
from hri_env_v2 import HriEnv_v2
import matplotlib.pyplot as plt
from ddpg import DDPG

model_path = './model'
hri_env = HriEnv_v2()
s_dim = 6
a_dim = hri_env.task_num
MAX_EP_STEPS = 200

ddpg_test = DDPG(a_dim, s_dim)
ddpg_test.load_weights(model_path)
# ddpg_test.eval()
# policy = lambda x: ddpg_test.choose_action(x)
# print(policy([0, 0, 0]))

s = hri_env.reset()
print('task_index')
print(hri_env.task_index)
print('x_d')
print(hri_env.x_d)
ep_reward = 0

b1_list = []
b2_list = []
b3_list = []
fh_x_list = []
fh_y_list = []
p_x_list = []
p_y_list = []
f_imp_x_list = []
f_imp_y_list = []
v_d_x = []
v_d_y = []
v_x = []
v_y = []
r1_list = []
r2_list = []
std_b = []

# a = [1. - hri_env.task_index, hri_env.task_index]
# a = [1., 0.]

# f1 = open('data/admittance_control/f_h.csv', mode='w')
# f1_writer = csv.writer(f1, delimiter=',')

for i in range(MAX_EP_STEPS):
    a = ddpg_test.choose_action(s, 0.)
    a = a.numpy().reshape(-1)
    s_, r = hri_env.step(a)
    s = s_
    ep_reward += r
    # print(a)
    b1_list.append(a[0])
    b2_list.append(a[1])
    b3_list.append(a[2])
    fh_x_list.append(hri_env.f_h[0])
    fh_y_list.append(hri_env.f_h[1])
    p_x_list.append(hri_env.ee_pos[0])
    p_y_list.append(hri_env.ee_pos[1])
    f_imp_x_list.append(hri_env.f_imp[0])
    f_imp_y_list.append(hri_env.f_imp[1])
    v_d_x.append(hri_env.dx_d_output[0])
    v_d_y.append(hri_env.dx_d_output[1])
    r1_list.append(hri_env.r_1)
    r2_list.append(hri_env.r_2)
    v_x.append(hri_env.ee_vel[0])
    v_y.append(hri_env.ee_vel[1])
    std_b.append(np.std(a))

    # if i == MAX_EP_STEPS-1:
    #     print('total reward: {:f}'.format(ep_reward))

plt.figure(1)
plt.plot(b1_list, label='task 1')
plt.plot(b2_list, label='task 2')
plt.plot(b3_list, label='task 3')
plt.legend()

plt.figure(2)
plt.plot(fh_x_list, label='fh x')
# plt.plot(fh_y_list, label='fh y')
plt.plot(f_imp_x_list, label='f_imp x')
# plt.plot(f_imp_y_list, label='f_imp y')
plt.legend()

plt.figure(3)
plt.plot(p_x_list, label='p x')
plt.plot(p_y_list, label='p y')
plt.legend()

plt.figure(4)
plt.plot(v_d_x, label='vd x')
# plt.plot(v_d_y, label='vd y')
plt.plot(v_x, label='v x')
# plt.plot(v_y, label='v y')
plt.legend()

plt.figure(5)
plt.plot(v_d_y, label='vd y')
plt.plot(v_y, label='v y')
plt.legend()

plt.figure(6)
plt.plot(r1_list, label='r1')
plt.plot(r2_list, label='r2')
plt.legend()

plt.figure(7)
plt.plot(std_b, label='std')
plt.legend()

plt.show()

