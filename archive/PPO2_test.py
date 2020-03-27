import gym
import os
import os.path as osp
import inspect
import sys

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.bench import Monitor

env = gym.make('MountainCarContinuous-v0')

agent_id = 3
os.mkdir(currentPath + '/logs/' + str(agent_id))
env = Monitor(env, 'logs/' + str(agent_id) + '/')

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(1e6))
model.save('logs/' + str(agent_id) + '/ppo2_mountain')

del model  # remove to demonstrate saving and loading
#
model = PPO2.load('logs/' + str(agent_id) + '/ppo2_mountain')

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()
