from gym.envs.registration import register

register(
    id='CarlaGymEnv-v95',
    entry_point='carla_gym.envs:CarlaGymEnv')
