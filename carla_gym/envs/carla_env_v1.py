# -*- coding: utf-8 -*-
"""
@author: Majid Moghadam
UCSC - ASL
"""

import copy
from modules import *
import gym
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.6.0"

        # simulation
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load('road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = 30  # km/h
        self.maxSpeed = 150
        self.maxAcc = 24.7608  # km/h.s OR 6.878 m/s^2 for Tesla model 3
        self.maxCte = 3
        self.maxTheta = math.pi / 2
        self.maxJerk = 1.5e2
        self.maxAngVelNorm = math.sqrt(2 * 180 ** 2) / 4  # maximum 180 deg/s around x and y axes;  /4 to end eps earlier and teach agent faster

        # frenet
        self.f_idx = 0
        self.init_s = None              # initial frenet s value - will be updated in reset function
        self.max_s = 3000            # max frenet s value available in global route
        self.track_length = 500      # distance to travel on s axis before terminating the episode. Must be <max_s-init_s.

        # RL
        self.low_state = np.array([-1, -1])
        self.high_state = np.array([1, 1])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)
        action_low = np.array([-1, -1])
        action_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])

        # instances
        self.ego = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.dt = 0.05
        self.acceleration_ = 0
        self.eps_rew = 0

        self.motionPlanner = None
        self.vehicleController = None

    def seed(self, seed=None):
        pass

    def step(self, action=None):
        # self.ego.set_autopilot(enabled=True)
        action = [0, 0]
        self.n_step += 1
        track_finished = False

        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]

        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed / 3.6, acc, psi,temp]
        #fpath = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])

        fpath = self.motionPlanner.run_step(ego_state, self.f_idx)[0]
        wps_to_go = len(fpath.t) - 2
        self.f_idx = 1

        speeds = []
        accelerations = []
        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        for _ in range(wps_to_go):
            self.f_idx += 1
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdSpeed = math.sqrt((fpath.s_d[self.f_idx]) ** 2 + (fpath.d_d[self.f_idx]) ** 2) * 3.6
            control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            self.ego.apply_control(control)               # apply control
            # print(fpath.s[self.f_idx], self.ego.get_transform().rotation.yaw)

            """
                    **********************************************************************************************************************
                    *********************************************** Draw Waypoints *******************************************************
                    **********************************************************************************************************************
            """
            # for j, path in enumerate(self.fplist):
            #     for i in range(len(path.t)):
            #         self.world_module.points_to_draw['path {} wp {}'.format(j, i)] = [carla.Location(x=path.x[i], y=path.y[i]), 'COLOR_SKY_BLUE_0']

            for i in range(len(fpath.t)):
                self.world_module.points_to_draw['path wp {}'.format(i)] = [carla.Location(x=fpath.x[i], y=fpath.y[i]), 'COLOR_ALUMINIUM_0']
            self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
            self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])

            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            speed_ = get_speed(self.ego)
            self.traffic_module.update_ego_s(fpath.s[self.f_idx])
            self.module_manager.tick()  # Update carla world
            if self.auto_render:
                self.render()

            speed = get_speed(self.ego)
            acc = (speed - speed_) / self.dt
            speeds.append(speed)
            accelerations.append(acc)
            s, d = fpath.s[self.f_idx], fpath.d[self.f_idx]

            # print('x:', fpath.x[self.f_idx], 'y:', fpath.y[self.f_idx], 'z:', fpath.z[self.f_idx])
            # print('x:', self.ego.get_location().x, 'y:', self.ego.get_location().y, 'z:', self.ego.get_location().z)
            # print(s)
            # print(100*'--')

            distance_traveled = s - self.init_s

            if distance_traveled >= self.track_length:
                track_finished = True
                break
            else:
                track_finished = False
        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        meanSpeed = np.mean(speeds)
        meanAcc = np.mean(accelerations)
        speed_n = (meanSpeed - self.targetSpeed) / self.targetSpeed  # -1<= speed_n <=1
        acc_n = meanAcc / (2 * self.maxAcc)  # -1<= acc_n <=1
        self.state = np.array([speed_n, acc_n])
        # print(self.state)
        # print(100 * '--')
        """
                **********************************************************************************************************************
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        w_speed = 10
        w_acc = 1 / 2
        e_speed = abs(self.targetSpeed - speed)
        r_speed = np.exp(-e_speed ** 2 / self.maxSpeed * w_speed)  # 0<= r_speed <= 1
        r_acc = np.exp(-abs(meanAcc) ** 2 / (2 * self.maxAcc) * w_acc) - 1  # -1<= r_acc <= 0
        r_laneChange = -abs(action[0])  # -1<= r_laneChange <= 0
        positives = r_speed
        negatives = (r_acc + r_laneChange) / 2
        reward = positives + negatives  # -1<= reward <=1
        # print(self.n_step, self.eps_rew)

        """
                **********************************************************************************************************************
                ********************************************* Episode Termination ****************************************************
                **********************************************************************************************************************
        """
        done = False
        if track_finished:
            # print('Finished the race')
            reward = 10
            done = True
            self.eps_rew += reward
            # print(self.n_step, self.eps_rew)
            return self.state, reward, done, {'reserved': 0}
        # if cte > self.maxCte:
        #     reward = -100
        #     # done = True
        #     self.eps_rew += reward
        #     # print(self.n_step, self.eps_rew)
        #     return self.state, reward, done, {'max index': self.max_idx_achieved}
        # if theta > self.maxTheta:
        #     reward = -100
        #     # done = True
        #     self.eps_rew += reward
        #     # print(self.n_step, self.eps_rew)
        #     return self.state, reward, done, {'max index': self.max_idx_achieved}
        # if w_norm > self.maxAngVelNorm:
        #     reward = -100
        #     # done = True
        #     self.eps_rew += reward
        #     # print(self.n_step, self.eps_rew)
        #     return self.state, reward, done, {'max index': self.max_idx_achieved}
        self.eps_rew += reward
        # print(self.n_step, self.eps_rew)
        return self.state, reward, done, {'reserved': 0}

    def reset(self):
        # self.state = np.array([0, 0], ndmin=2)
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        self.traffic_module.reset(self.init_s)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d)

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.state = np.array([0 for _ in range(self.observation_space.shape[0])])  # initialize state vector
        self.module_manager.tick()
        return np.array(self.state)

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height, max_s=self.max_s, track_length=self.track_length)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager, max_s=self.max_s, track_length=self.track_length)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        if self.world_module.dt is not None:
            self.dt = self.world_module.dt
        else:
            self.dt = 0.05

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y, wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

        self.motionPlanner = MotionPlanner(dt=self.dt, targetSpeed=self.targetSpeed / 3.6)

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.vehicleController = VehiclePIDController(self.ego)
        self.module_manager.tick()  # Update carla world

        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
