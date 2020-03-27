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
from stable_baselines.ddpg.policies import MlpPolicy as DDPGPolicy
from stable_baselines.common.policies import MlpPolicy as CommonMlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO

import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v95')
    parser.add_argument('--alg', help='RL algorithm', type=str, default='ddpg')
    parser.add_argument('--action_noise', help='Action noise', type=float, default=0.5)
    parser.add_argument('--param_noise_stddev', help='Param noise', type=float, default=0.0)
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=None),
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
    args.num_timesteps = int(args.num_timesteps)
    print('Env is starting')
    env = gym.make(args.env)

    if args.agent_id is not None:
        os.mkdir(currentPath + '/logs/agent_{}/'.format(args.agent_id))
        os.mkdir(currentPath + '/logs/agent_{}/models/'.format(args.agent_id))
        env = Monitor(env, 'logs/agent_{}/'.format(args.agent_id), info_keywords=('max index',))
    else:
        env = Monitor(env, 'logs/', info_keywords=('max index',))

    env.begin_modules(args)

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]

    # model save/load directory
    if args.agent_id is not None:
        model_dir = '/logs/agent_{}/models/{}_final_model'.format(args.agent_id, args.alg)
    else:
        model_dir = '/logs/{}_final_model'.format(args.alg)

    if not args.test:
        if args.alg == 'ddpg':
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=args.action_noise * np.ones(n_actions))

            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(args.param_noise_stddev),
                                                 desired_action_stddev=float(args.param_noise_stddev))
            model = DDPG(DDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,
                         render=args.play)
        elif args.alg == 'ppo2':
            model = PPO2(CommonMlpPolicy, env, verbose=1)
        elif args.alg == 'trpo':
            model = TRPO(CommonMlpPolicy, env, verbose=1)
        else:
            print(args.alg)
            raise Exception('Algorithm name is not defined!')

        print('Model is Created')
        try:
            print('Training Started')
            model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, agent_id=args.agent_id)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            # save model even if training fails because of an error
            model.save(model_dir)
            env.destroy()
            print('model has been saved.')

    else:
        if args.alg == 'ddpg':
            model = DDPG.load(model_dir)
            model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                              sigma=np.zeros(n_actions))
            model.param_noise = None
        elif args.alg == 'ppo2':
            model = PPO2.load(model_dir)
        elif args.alg == 'trpo':
            model = TRPO.load(model_dir)
        else:
            print(args.alg)
            raise Exception('Algorithm name is not defined!')

        print('Model is loaded')
        try:
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
        finally:
            env.destroy()
