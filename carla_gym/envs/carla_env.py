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
        self.__version__ = "9.5.0"
        self.n_step = 0
        self.point_cloud = []  # race waypoints (center lane)
        self.LOS = 20  # line of sight, i.e. number of cloud points to interpolate road curvature
        self.poly_deg = 3  # polynomial degree to fit the road curvature points
        self.targetSpeed = 80  # km/h
        self.maxSpeed = 150
        self.maxDist = 5
        self.accum_speed_e = 0

        self.low_state = np.append([-float('inf') for _ in range(self.poly_deg + 1)], [-1, -1])
        self.high_state = np.append([float('inf') for _ in range(self.poly_deg + 1)], [1, 1])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)
        action_low = np.array([-1/2, -20, -10])  # action = [throttle , WPb_x (m), WPb_y (m)]
        action_high = np.array([1/2, 20, 10])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])
        self.module_manager = None
        self.world_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None      # ego initial transform to recover at each episode

    def seed(self, seed=None):
        pass

    def closest_point_cloud_index(self, ego_pos):
        # find closest point in point cloud
        min_dist = None
        min_idx = 0
        for idx, point in enumerate(self.point_cloud):
            dist = euclidean_distance([ego_pos.x, ego_pos.y, ego_pos.z], [point.x, point.y, point.z])
            if min_dist is None:
                min_dist = dist
            else:
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx

        return min_idx, min_dist

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

        if len(self.point_cloud) - idx <= 50:
            track_finished = True
        else:
            track_finished = False

        # update the curvature points window (points in body frame)
        curvature_points = self.update_curvature_points(ego_transform, close_could_idx=idx)

        # fit a polynomial to the curvature points window
        c = np.polyfit([p[0] for p in curvature_points], [p[1] for p in curvature_points], self.poly_deg)
        # c: coefficients in decreasing power

        if draw_poly:
            poly = np.poly1d(c)
            psi = math.radians(ego_transform.rotation.yaw)
            for x in range(1, 50, 2):
                pi = self.world_module.body_to_inertial_frame(x, poly(x), psi)
                self.world_module.points_to_draw['poly fit {}'.format(x)] = carla.Location(x=pi[0], y=pi[1])

        return c, dist, track_finished

    def step(self, action=None):
        self.n_step += 1
        action[0] += 1/2
        action[1] += 20  # only move forward
        # Apply action
        # action = None
        self.module_manager.tick()  # Update carla world and lat/lon controllers
        speed = self.control_module.tick(action)  # apply control

        # Calculate observation vector
        ego_transform = self.world_module.hero_actor.get_transform()
        c, dist, track_finished = self.interpolate_road_curvature(ego_transform, draw_poly=False)
        yaw_norm = ego_transform.rotation.yaw / 180
        speed_e = (self.targetSpeed - speed) / self.maxSpeed  # normalized speed error
        self.accum_speed_e += speed_e
        self.state = np.append(c, [yaw_norm, speed_e])

        # Reward function
        speed_e_r = abs(self.targetSpeed - speed) / self.maxSpeed  # normalized speed error
        dist_r = dist / self.maxDist
        reward = -1 * (speed_e_r + dist_r)  # -1<= reward <= 1

        # Episode
        done = False
        if track_finished:
            print('Finished the race')
            reward = 1000.0
            done = True
            return self.state, reward, done, {}
        if dist >= self.maxDist:
            reward = -5.0
            done = True
            return self.state, reward, done, {}

        if (self.n_step >= 200) and (self.accum_speed_e/self.n_step) > 0.5:     # terminate if agent did not move much
            reward = -5.0
            done = True
            return self.state, reward, done, {}

        return self.state, reward, done, {}

    def reset(self):
        # self.state = np.array([0, 0], ndmin=2)
        # Set ego transform to its initial form
        self.world_module.hero_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.world_module.hero_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
        self.world_module.hero_actor.set_transform(self.init_transform)

        self.accum_speed_e = 0
        self.n_step = 0     # initialize episode steps count
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])      # initialize state vector
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
        # We do not register control module bc we want to tick control separately
        # self.module_manager.register_module(self.control_module)

        # Start Modules
        self.module_manager.start_modules()
        self.control_module.start()
        self.module_manager.tick()  # Update carla world and lat/lon controllers
        self.control_module.tick()  # apply control

        self.init_transform = self.world_module.hero_actor.get_transform()

        distance = 0
        for i in range(1520):
            distance += 2
            wp = self.world_module.town_map.get_waypoint(self.world_module.hero_actor.get_location(),
                                                         project_to_road=True).next(distance=distance)[0]
            self.point_cloud.append(wp.transform.location)

            # To visualize point clouds
            # self.world_module.points_to_draw['wp {}'.format(wp.id)] = wp.transform.location

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self, keepWorld=False):
        # print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
