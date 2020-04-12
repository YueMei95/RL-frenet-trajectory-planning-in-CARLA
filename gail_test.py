import gym
import numpy as np
import os.path as osp
import inspect
import sys

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

from stable_baselines import DDPG
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import MlpPolicy as DDPGPolicy

from stable_baselines.bench import Monitor

# Generate expert trajectories (train expert)
env = gym.make('MountainCarContinuous-v0')
# n_actions = env.action_space.shape[-1]  # the noise objects for DDPG
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
#                                             sigma=0.5 * np.ones(n_actions))
#
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.0),
#                                      desired_action_stddev=float(0.0))
# model = DDPG(DDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=False)
#
# model.load('logs/agent_200/models/ddpg_final_model')
#
# generate_expert_traj(model, 'logs/agent_200/models/expert_car', n_timesteps=0, n_episodes=500)
#
# # Load the expert dataset
# dataset = ExpertDataset(expert_path='logs/agent_200/models/expert_car.npz', traj_limitation=-1, verbose=1)
#
# env = Monitor(env, 'logs/agent_200/', )  # logging monitor
# model = GAIL('MlpPolicy', env, dataset, verbose=1)
# # Note: in practice, you need to train for 1M steps to have a working policy
# model.learn(total_timesteps=1e6)
# model.save("logs/agent_200/models/gail_car")

# del model # remove to demonstrate saving and loading

model = GAIL.load("logs/agent_200/models/gail_car")

obs = env.reset()
while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()
  if dones:
      env.reset()