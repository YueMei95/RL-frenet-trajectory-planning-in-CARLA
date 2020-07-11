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
from stable_baselines.common.policies import MlpLstmPolicy as CommonMlpLstmPolicy
from stable_baselines.common.policies import CnnPolicy as CommonCnnPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines import A2C

import argparse
import git


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v95')
    parser.add_argument('--alg', help='RL algorithm', type=str, default='ddpg')
    parser.add_argument('--action_noise', help='Action noise', type=float, default=0.5)
    parser.add_argument('--param_noise_stddev', help='Param noise', type=float, default=0.0)
    parser.add_argument('--log_interval', help='Log interval (model)', type=int, default=100)
    parser.add_argument('--agent_id', type=int, default=None),
    parser.add_argument('--num_timesteps', type=float, default=1e7),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp')
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--test_model', help='test model file name', type=str, default='')
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    args = parser.parse_args()

    # correct default test_model arg
    if args.test_model == '':
        args.test_model = '{}_final_model'.format(args.alg)

    # visualize all test scenarios
    if args.test:
        args.play_mode = True

    args.num_timesteps = int(args.num_timesteps)
    return args



if __name__ == '__main__':
    args = parse_args()
    print('Env is starting')
    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)
    n_actions = env.action_space.shape[-1]  # the noise objects for DDPG

    # --------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------Training----------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    
    policy = {'mlp': CommonMlpPolicy, 'lstm': CommonMlpLstmPolicy, 'cnn': CommonCnnPolicy}

    if not args.test:  # training
        if args.agent_id is not None:
            os.mkdir(currentPath + '/logs/agent_{}/'.format(args.agent_id))                             # create agent_id folder
            os.mkdir(currentPath + '/logs/agent_{}/models/'.format(args.agent_id))
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
            env = Monitor(env, 'logs/agent_{}/'.format(args.agent_id))    # logging monitor

            repo = git.Repo(search_parent_directories=False)
            commit_id = repo.head.object.hexsha
            with open('logs/agent_{}/reproduction_info.txt'.format(args.agent_id), 'w') as f:  # Use file to refer to the file object
                f.write('Git commit id: {}\n\n'.format(commit_id))
                f.write('Program arguments:\n\n{}'.format(args))
                f.close()
        else:
            save_path = 'logs/'
            env = Monitor(env, 'logs/', info_keywords=('reserved',))                                   # logging monitor
        model_dir = save_path + '{}_final_model'.format(args.alg)                                       # model save/load directory

        if args.alg == 'ddpg':
            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                        sigma=args.action_noise * np.ones(n_actions))

            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(args.param_noise_stddev),
                                                 desired_action_stddev=float(args.param_noise_stddev))
            model = DDPG(DDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
        elif args.alg == 'ppo2':
            model = PPO2(policy[args.network], env, verbose=1, model_dir=save_path)
        elif args.alg == 'trpo':
            model = TRPO(policy[args.network], env, verbose=1, model_dir=save_path)
        elif args.alg =='a2c':
            model = A2C(policy[args.network], env, verbose=1)
        else:
            print(args.alg)
            raise Exception('Algorithm name is not defined!')

        print('Model is Created')
        try:
            print('Training Started')
            if args.alg == 'ddpg':
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval, save_path=save_path)
            else:
                model.learn(total_timesteps=args.num_timesteps, log_interval=args.log_interval)
        finally:
            print(100 * '*')
            print('FINISHED TRAINING; saving model...')
            print(100 * '*')
            # save model even if training fails because of an error
            model.save(model_dir)
            env.destroy()
            print('model has been saved.')

    # --------------------------------------------------------------------------------------------------------------------"""
    # ------------------------------------------------Test----------------------------------------------------------------"""
    # --------------------------------------------------------------------------------------------------------------------"""

    else:  # test
        if args.agent_id is not None:
            save_path = 'logs/agent_{}/models/'.format(args.agent_id)
        else:
            save_path = 'logs/'
        model_dir = save_path + args.test_model  # model save/load directory

        if args.alg == 'ddpg':
            model = DDPG.load(model_dir)
            model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                              sigma=np.zeros(n_actions))
            model.param_noise = None
        elif args.alg == 'ppo2':
            model = PPO2.load(model_dir)
        elif args.alg == 'trpo':
            model = TRPO.load(model_dir)
        elif args.alg == 'a2c':
            model = A2C.load(model_dir)
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
