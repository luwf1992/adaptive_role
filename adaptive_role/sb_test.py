from matplotlib import pyplot as plt
from stable_baselines3 import DDPG, TD3, SAC, PPO
import gym

env_id = 1
env = gym.make("NewHriEnv-v" + str(env_id))
model = TD3.load("./sb3/model/td3_hrienv-v1.zip", env=env)
MAX_EP_STEPS = 200

fh_x_list = []
fh_y_list = []
fh_z_list = []
p_x_list = []
p_y_list = []
f_imp_x_list = []
f_imp_y_list = []
f_imp_z_list = []
v_x = []
v_y = []
r_list = []

obs = env.reset()
print('task_index')
print(env.task_index)
print('x_d')
print(env.x_d)
ep_reward = 0
for i in range(MAX_EP_STEPS):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, _, _ = env.step(action)
    fh_x_list.append(env.f_h[0])
    fh_y_list.append(env.f_h[1])
    fh_z_list.append(env.f_h[2])
    p_x_list.append(env.ee_pos[0])
    p_y_list.append(env.ee_pos[1])
    f_imp_x_list.append(action[0])
    f_imp_y_list.append(action[1])
    f_imp_z_list.append(action[2])
    v_x.append(env.ee_vel[0])
    v_y.append(env.ee_vel[1])
    r_list.append(rewards)

plt.figure(1)
plt.plot(fh_x_list, label='fh x')
plt.plot(fh_y_list, label='fh y')
plt.plot(f_imp_x_list, label='f_imp x')
plt.plot(f_imp_y_list, label='f_imp y')
plt.legend()

plt.figure(2)
plt.plot(p_x_list, label='p x')
plt.plot(p_y_list, label='p y')
plt.legend()

plt.figure(3)
plt.plot(v_x, label='v x')
plt.plot(v_y, label='v y')
plt.legend()

plt.figure(4)
plt.plot(r_list, label='r')
plt.legend()

plt.figure(5)
plt.plot(fh_z_list, label='fh z')
plt.plot(f_imp_z_list, label='f_imp z')
plt.legend()

plt.show()















