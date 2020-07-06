# -*- coding: utf-8 -*-
"""
@author: Majid Moghadam
UCSC - ASL
"""

import copy
from modules import *
import gym
import time
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    # print('{}--{}--{}'.format(f_idx,closest_wp_index,inertial_to_body_frame(ego_location, fpath.x[closest_wp_index+ f_idx],fpath.x[closest_wp_index + f_idx],ego_state[2] )[0]))
    return f_idx + closest_wp_index


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.9.2"

        # simulation
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = 50 / 3.6  # m/s
        self.planner_speed_range = [20/3.6, 55/3.6]
        self.traffic_speed_range = [30/3.6, 40/3.6]
        self.maxSpeed = 150 / 3.6  # m/s
        self.maxAcc = 6.878  # m/s^2 or 24.7608 km/h.s for Tesla model 3
        self.LANE_WIDTH = 3.5  # lane width [m]
        self.N_INIT_CARS = 15   # number of other actors

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = 3000  # max frenet s value available in global route
        self.track_length = 500  # distance to travel on s axis before terminating the episode. Must be less than self.max_s - 50
        self.lookback = 30
        self.loop_break = 50    # must be greater than loop_break

        # RL
        self.low_state = np.array([[-1 for _ in range(self.lookback)], [-1 for _ in range(self.lookback)]])
        self.high_state = np.array([[1 for _ in range(self.lookback)], [1 for _ in range(self.lookback)]])
        self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
                                                dtype=np.float32)
        action_low = np.array([-1, -1])
        action_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.array([[0 for _ in range(self.observation_space.shape[1])], [0 for _ in range(self.observation_space.shape[1])]])

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
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp,self.max_s]
        fpath = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])
        wps_to_go = len(fpath.t) - 3    # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1

        speeds = []
        accelerations = []
        actors_norm_s = []    # relative frenet s value wrt ego
        actors_norm_d = []    # relative frenet d value wrt ego
        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()

        # follows path until end of WPs for max 1.8seconds
        loop_counter = 0
        while self.f_idx < wps_to_go and elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5:
            loop_counter += 1
            # for _ in range(wps_to_go):
            # self.f_idx += 1
            ego_location = [self.ego.get_location().x, self.ego.get_location().y, math.radians(self.ego.get_transform().rotation.yaw)]
            self.f_idx = closest_wp_idx(ego_location, fpath, self.f_idx)
            cmdSpeed = math.sqrt((fpath.s_d[self.f_idx]) ** 2 + (fpath.d_d[self.f_idx]) ** 2)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # IDM for ego: comment out for RL training.
            # vehicle_ahead = self.world_module.los_sensor.get_vehicle_ahead()
            # cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)
            # nextWP = self.world_module.town_map.get_waypoint(self.ego.get_location(), project_to_road=True).next(distance=10)[0]
            # cmdWP = [nextWP.transform.location.x, nextWP.transform.location.y]

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control
            # print(fpath.s[self.f_idx], self.ego.get_transform().rotation.yaw)

            """
                    **********************************************************************************************************************
                    *********************************************** Draw Waypoints *******************************************************
                    **********************************************************************************************************************
            """
            # for j, path in enumerate(self.fplist):
            #     for i in range(len(path.t)):
            #         self.world_module.points_to_draw['path {} wp {}'.format(j, i)] = [carla.Location(x=path.x[i], y=path.y[i]), 'COLOR_SKY_BLUE_0']
            if self.world_module.args.play_mode != 0:
                for i in range(len(fpath.t)):
                    self.world_module.points_to_draw['path wp {}'.format(i)] = [carla.Location(x=fpath.x[i], y=fpath.y[i]),
                                                                                'COLOR_ALUMINIUM_0']
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            speed_ = get_speed(self.ego)    # speed in previous tick
            self.module_manager.tick()  # Update carla world
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()

            speed = get_speed(self.ego)
            acc = (speed - speed_) / self.dt
            speeds.append(speed)
            accelerations.append(acc)
            ego_s, ego_d = fpath.s[self.f_idx], fpath.d[self.f_idx]
            norm_s = [0 for _ in range(self.N_INIT_CARS)]
            norm_d = [0 for _ in range(self.N_INIT_CARS)]
            for i, actor in enumerate(self.traffic_module.actors_batch):
                act_s, act_d = actor['Frenet State']
                norm_s[i] = (act_s - ego_s) / self.max_s
                norm_d[i] = (act_d / (2*self.LANE_WIDTH))
            actors_norm_s.append(norm_s)
            actors_norm_d.append(norm_d)

            # loop breakers:
            if any(collision_hist):
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True
                break
            if loop_counter >= self.loop_break:
                break
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
        speeds_vec = (np.array(speeds) - self.targetSpeed)/self.targetSpeed
        accelerations_vec = np.array(accelerations)/(2*self.maxAcc)
        speeds_acc = np.concatenate((speeds_vec, accelerations_vec), axis=0)
        speeds_acc = speeds_acc.reshape(2, -1)
        self.state = speeds_acc[:, -self.lookback:]                                                          
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
        if collision:
            # print('Collision happened!')
            reward = -10
            done = True
            self.eps_rew += reward
            # print(self.n_step, self.eps_rew)
            return self.state, reward, done, {'reserved': 0}
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
        self.vehicleController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0

        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        # self.state = np.array([0 for _ in range(self.observation_space.shape[0])])  # initialize state vector
        self.state = np.array([[0 for _ in range(self.observation_space.shape[1])], [0 for _ in range(self.observation_space.shape[1])]]) 
        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        return np.array(self.state)

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height, max_s=self.max_s, track_length=self.track_length)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager, N_INIT_CARS=self.N_INIT_CARS, max_s=self.max_s,
                                             track_length=self.track_length,
                                             min_speed=self.traffic_speed_range[0], max_speed=self.traffic_speed_range[1])
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
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

        self.motionPlanner = MotionPlanner(dt=self.dt, targetSpeed=self.targetSpeed,
                                           speed_min=self.planner_speed_range[0], speed_max=self.planner_speed_range[1])

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.vehicleController = VehiclePIDController(self.ego)
        self.IDM = IntelligentDriverModel(self.ego, self.dt)

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
            self.traffic_module.destroy()
