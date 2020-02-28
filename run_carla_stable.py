import gym
import carla_gym
import numpy as np
import os.path as osp
import os
import inspect
import sys

currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, currentPath + '/agents/stable_baselines/')

from stable_baselines.bench import Monitor
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG




import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v95')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1',
                        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720',
                        help='window resolution (default: 1280x720)')
    args, _ = parser.parse_known_args(sys.argv)
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Env is starting')
    env = gym.make(args.env)
    env = Monitor(env, 'logs/')

    env.begin_modules(args)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=args.play)
    print('Model is Created')
    if not args.test:
        try:
            print('Training Started')
            model.learn(total_timesteps=args.num_timesteps)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            model.save('models/ddpg_carla')  # save model even if training fails because of an error
            env.destroy()

    if args.test:
        model = DDPG.load('models/ddpg_carla')

        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                env.render()
        finally:
            env.destroy()
