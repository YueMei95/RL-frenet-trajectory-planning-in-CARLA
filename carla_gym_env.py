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


class CarlaEnv:
    def __init__(self, args):
        self.module_manager = ModuleManager()
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=2.0, module_manager=self.module_manager)
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
        self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
        self.control_module = ModuleControl(MODULE_CONTROL, module_manager=self.module_manager)

        # Register Modules
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.hud_module)
        self.module_manager.register_module(self.input_module)
        self.module_manager.register_module(self.control_module)

        self.module_manager.start_modules()

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(1,), dtype=np.float32)

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)

        self.reset()

    def seed(self, seed=None):
        pass

    def step(self, action=0):
        self.world_module.clock.tick_busy_loop()
        self.module_manager.tick(self.world_module.clock)
        reward = np.array([0.0])
        done = np.array([False])
        self.state = np.array([0, 0], ndmin=2)
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([0, 0], ndmin=2)
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def _height(self, xs):
        pass

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)
        pass

    def close(self):
        if self.world_module is not None:
            self.world_module.destroy()
