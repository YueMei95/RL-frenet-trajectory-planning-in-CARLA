from gym.envs.registration import register

register(
    id='CarlaGymEnv-v95',
    entry_point='carla_gym.envs:CarlaGymEnv')

register(
    id='CarlaGymEnv-v0',
    entry_point='carla_gym.envs:CarlaGymEnv_v0')