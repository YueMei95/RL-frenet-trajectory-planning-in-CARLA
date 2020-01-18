import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)

from carla_gym_env import CarlaEnv

import argparse


def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(
        description='CARLA No Rendering Mode Visualizer')
    argparser.add_argument(
        '--carla_host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--carla_port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--carla_res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()
    args.description = argparser.description
    args.width, args.height = [int(x) for x in args.carla_res.split('x')]

    env = CarlaEnv(args)
    try:
        while True:
            env.step()
            env.render()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        env.close()


if __name__ == '__main__':
    main()
