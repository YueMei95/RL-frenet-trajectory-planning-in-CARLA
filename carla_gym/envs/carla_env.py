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


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.6.0"
        self.n_step = 0
        self.point_cloud = []  # race waypoints (center lane)
        self.min_idx = 0   # keep track of last closest idx in point cloud to reduce search space to find closest idx
        self.max_idx_achieved = 0
        self.max_idx = 50       # max idx to finish the episode
        self.LOS = 15  # line of sight, i.e. number of cloud points to interpolate road curvature
        self.poly_deg = 3  # polynomial degree to fit the road curvature points
        self.targetSpeed = 50  # km/h
        self.maxSpeed = 150
        self.maxCte = 3
        self.maxTheta = math.pi/2
        self.maxJerk = 1.5e2
        self.maxAngVelNorm = math.sqrt(2 * 180 ** 2) / 4  # maximum 180 deg/s around x and y axes;  /4 to end eps earlier and teach agent faster

        self.low_state = np.append([-float('inf') for _ in range(self.poly_deg+1)], [0, 0, -180, -180, -180])
        self.high_state = np.append([float('inf') for _ in range(self.poly_deg+1)], [self.maxSpeed, self.maxSpeed, 180, 180, 180])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)
        action_low = np.array([-1])  # steering
        action_high = np.array([1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])
        self.module_manager = None
        self.world_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.dt = 0.05
        self.acceleration_ = 0
        self.eps_rew = 0

    def seed(self, seed=None):
        pass

    def closest_point_cloud_index(self, ego_pos):
        # find closest point in point cloud
        min_dist = None
        i = max(0, self.min_idx - 5)    # window size = 10
        j = i + 10
        min_idx = 0
        for idx, point in enumerate(self.point_cloud[i:j]):
            dist = euclidean_distance([ego_pos.x, ego_pos.y, ego_pos.z], [point.x, point.y, point.z])
            if min_dist is None:
                min_dist = dist
            else:
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
        self.min_idx = i + min_idx
        if self.min_idx > self.max_idx_achieved:
            self.max_idx_achieved = self.min_idx
        return self.min_idx, min_dist

    # This function needs to be optimized in terms of time complexity
    # remove closest point add new one to end instead of calculation LOS points at every iteration
    def update_curvature_points(self, ego_transform, close_could_idx, draw_points=False):
        # transfer could points to body frame
        psi = math.radians(ego_transform.rotation.yaw)
        curvature_points = [
            self.world_module.inertial_to_body_frame(self.point_cloud[i].x, self.point_cloud[i].y, psi)
            for i in range(close_could_idx, close_could_idx + self.LOS)]

        if draw_points:
            for i, point in enumerate(curvature_points):
                pi = self.world_module.body_to_inertial_frame(point[0], point[1], psi)
                self.world_module.points_to_draw['curvature cloud {}'.format(i)] = carla.Location(x=pi[0], y=pi[1])

        return curvature_points

    def interpolate_road_curvature(self, ego_transform, draw_poly=False):
        # find the index of the closest point cloud to the ego
        idx, dist = self.closest_point_cloud_index(ego_transform.location)

        # if len(self.point_cloud) - idx <= 50:
        if idx >= self.max_idx:
            track_finished = True
        else:
            track_finished = False

        # update the curvature points window (points in body frame)
        curvature_points = self.update_curvature_points(ego_transform, close_could_idx=idx, draw_points=False)

        # fit a polynomial to the curvature points window
        c = np.polyfit([p[0] for p in curvature_points], [p[1] for p in curvature_points], self.poly_deg)
        # c: coefficients in decreasing power

        if draw_poly:
            poly = np.poly1d(c)
            psi = math.radians(ego_transform.rotation.yaw)
            for x in range(-15, 50, 1):
                pi = self.world_module.body_to_inertial_frame(x, poly(x), psi)
                self.world_module.points_to_draw['poly fit {}'.format(x)] = carla.Location(x=pi[0], y=pi[1])

        return c, track_finished

    def step(self, action=None):
        self.n_step += 1

        # Apply action
        # action = None

        self.module_manager.tick()  # Update carla world and lat/lon controllers

        speed = self.control_module.tick(action=action, targetSpeed=self.targetSpeed)  # apply control
        # print(speed)

        # Calculate observation vector
        ego_transform = self.world_module.hero_actor.get_transform()
        c, track_finished = self.interpolate_road_curvature(ego_transform, draw_poly=False)
        w = self.world_module.hero_actor.get_angular_velocity()         # angular velocity
        self.state = np.append(c, [speed, self.targetSpeed, w.x, w.y, w.z])
        # print(self.state)

        # reward function
        cte = abs(c[-1])                 # cross track error
        theta = abs(math.atan(c[-2]))    # heading error wrt road curvature in radians. c[-2] is the slope
        w_norm = math.sqrt(sum([w.x ** 2 + w.y ** 2 + w.z ** 2]))
        reward = 1 - (cte/self.maxCte + theta/self.maxTheta + w_norm/self.maxAngVelNorm)/3
        self.eps_rew += reward
        # print(self.n_step, self.eps_rew)
        # print(reward)

        # Episode
        done = False
        if track_finished:
            # print('Finished the race')
            reward = 100
            done = True
            return self.state, reward, done, {'max index': self.max_idx_achieved}
        if cte > self.maxCte:
            reward = -1.0
            done = True
            return self.state, reward, done, {'max index': self.max_idx_achieved}
        if theta > self.maxTheta:
            reward = -1.0
            done = True
            return self.state, reward, done, {'max index': self.max_idx_achieved}
        if w_norm > self.maxAngVelNorm:
            reward = -1.0
            done = True
            return self.state, reward, done, {'max index': self.max_idx_achieved}
        return self.state, reward, done, {'max index': self.max_idx_achieved}

    def reset(self):
        # self.state = np.array([0, 0], ndmin=2)
        # Set ego transform to its initial form
        self.world_module.hero_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.world_module.hero_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.world_module.hero_actor.set_transform(self.init_transform)

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.min_idx = 0
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])  # initialize state vector
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def begin_modules(self, args):
        self.module_manager = ModuleManager()
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        if args.play:
            width, height = [int(x) for x in args.carla_res.split('x')]
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)
        self.control_module = ModuleControl(MODULE_CONTROL, module_manager=self.module_manager)
        # We do not register control module bc we want to tick control separately with different arguments
        # self.module_manager.register_module(self.control_module)

        # Start Modules
        self.module_manager.start_modules()
        self.control_module.start()
        self.module_manager.tick()  # Update carla world and lat/lon controllers
        self.control_module.tick()  # apply control

        self.init_transform = self.world_module.hero_actor.get_transform()
        if self.world_module.dt is not None:
            self.dt = self.world_module.dt
        else:
            self.dt = 0.05
        distance = 0
        print('Spawn the actor in: ', self.world_module.hero_actor.get_location())

        for i in range(1520):
            distance += 2
            wp = self.world_module.town_map.get_waypoint(self.world_module.hero_actor.get_location(),
                                                         project_to_road=True).next(distance=distance)[0]
            self.point_cloud.append(wp.transform.location)

            # To visualize point clouds
            # self.world_module.points_to_draw['wp {}'.format(wp.id)] = wp.transform.location

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
