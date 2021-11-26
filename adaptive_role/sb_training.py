import gym
import numpy as np
import torch as th

from stable_baselines3 import DDPG, TD3, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

env_id = 1
env = gym.make("NewHriEnv-v" + str(env_id))
check_env(env)

rl_algorithm = 'TD3'

log_dir = rl_algorithm + '_log_env' + str(env_id)
env = Monitor(env, log_dir)
savemodelcallback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
policy_kwargs = dict(activation_fn=th.nn.Hardshrink)

if rl_algorithm == 'DDPG':
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, buffer_size=100000, learning_starts=100, gamma=0.98,
                learning_rate=0.001, action_noise=None, verbose=1, batch_size=128,
                tau=0.01, tensorboard_log='./sb3/hri-v1_tensorboard/')
    model.learn(total_timesteps=100000, log_interval=10)
    model.save("./sb3/model/ddpg_hrienv-v1_2")

if rl_algorithm == 'TD3':
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0. * np.ones(n_actions))

    model = TD3("MlpPolicy", env, buffer_size=10000, learning_starts=10000, gamma=0.98,
                learning_rate=0.001, action_noise=None, verbose=1, batch_size=32,
                tau=0.01, policy_kwargs=policy_kwargs, tensorboard_log='./sb3/td3_hriv'+str(env_id)+'_tensorboard/')
    model.learn(total_timesteps=200000, log_interval=10, callback=savemodelcallback)
    model.save('./sb3/model/td3_hrienv-v'+str(env_id))

if rl_algorithm == 'SAC':
    model = SAC("MlpPolicy", env, buffer_size=100000, learning_starts=100, gamma=0.98,
                learning_rate=0.001, action_noise=None, verbose=1, batch_size=128,
                tau=0.01, policy_kwargs=policy_kwargs, tensorboard_log='./sb3/hri-v1_tensorboard/')
    model.learn(total_timesteps=20000, log_interval=10)
    model.save("./sb3/model/sac_hrienv-v1")

if rl_algorithm == 'PPO':
    env = make_vec_env("HriEnv-v1", n_envs=8)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log='./sb3/ppo_hriv1_tensorboard/')
    model.learn(total_timesteps=25000)
    model.save("./sb3/model/ppo_hrienv-v1")

















