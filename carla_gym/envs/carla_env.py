# -*- coding: utf-8 -*-
"""
@author: Majid Moghadam
UCSC - ASL
"""

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_CONTROL = 'CONTROL'

import numpy as np
from modules import *
import gym


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.5.0"
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(1,), dtype=np.float32)
        self.state = np.array([0, 0])
        self.module_manager = None
        self.world_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None

    def seed(self, seed=None):
        pass

    def step(self, action=0):
        self.module_manager.tick()  # Update carla world and lat/lon controllers
        reward = np.array([0.0])
        done = np.array([False])
        # ego_transform = self.world_module.hero_actor.get_transform()

        # psi = math.radians(ego_transform.rotation.yaw)

        # xwp, ywp = 5, 5
        # wpi = self.world_module.body_to_inertial_frame(xwp, ywp, psi)
        # self.world_module.points_to_draw['testWP'] = carla.Location(x=wpi[0], y=wpi[1])
        # print('x= {0:0.6f},  y= {1:0.6f}'.format(x, y))

        self.state = np.array([0, 0])
        return self.state, reward, done, {}

    def reset(self):
        # self.state = np.array([0, 0], ndmin=2)
        self.state = np.array([0, 0])
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def begin_modules(self, args):
        self.module_manager = ModuleManager()
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=2.0, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        if args.play:
            width, height = [int(x) for x in args.carla_res.split('x')]
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)
        self.control_module = ModuleControl(MODULE_CONTROL, module_manager=self.module_manager)
        self.module_manager.register_module(self.control_module)

        # Start Modules
        self.module_manager.start_modules()

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
