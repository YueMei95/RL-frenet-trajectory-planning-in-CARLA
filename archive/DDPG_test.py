import gym
import numpy as np
import os
import os.path as osp
import inspect
import sys

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor

env = gym.make('MountainCarContinuous-v0')

agent_id = 2
# os.mkdir(currentPath + '/logs/' + str(agent_id))
# env = Monitor(env, 'logs/' + str(agent_id) + '/')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0) * np.ones(n_actions))
#
# model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=False)
# model.learn(total_timesteps=1e6)
# model.save('logs/' + str(agent_id) + '/ddpg_mountain')

# del model # remove to demonstrate saving and loading

model = DDPG.load('logs/' + str(agent_id) + '/ddpg_mountain')
model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=np.zeros(n_actions))
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
    env.render()