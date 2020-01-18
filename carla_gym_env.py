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


class CarlaEnv:
    def __init__(self, args):
        self.module_manager = ModuleManager()
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=2.0, module_manager=self.module_manager)
        self.hud_module = ModuleHUD(MODULE_HUD, args.width, args.height, module_manager=self.module_manager)
        self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
        self.control_module = ModuleControl(MODULE_CONTROL, module_manager=self.module_manager)

        # Register Modules
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.hud_module)
        self.module_manager.register_module(self.input_module)
        self.module_manager.register_module(self.control_module)

        self.module_manager.start_modules()

        self.state = 0

    def seed(self, seed=None):
        pass

    def step(self, action=0):
        self.world_module.clock.tick_busy_loop()
        self.module_manager.tick(self.world_module.clock)
        reward = 0
        done = 0
        return self.state, reward, done, {}

    def reset(self):
        pass

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
