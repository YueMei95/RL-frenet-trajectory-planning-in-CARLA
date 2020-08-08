"""
@author: Majid Moghadam
UCSC - ASL
"""

import gym
import time
from tools.modules import *
from config import cfg
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP', \
                    'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']


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
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.maxAcc = float(cfg.GYM_ENV.MAX_ACC)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)

        # RL
        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.low_state = np.array([[-1 for _ in range(self.look_back)] for _ in range(16)])
            self.high_state = np.array([[1 for _ in range(self.look_back)] for _ in range(16)])
        else:
            self.low_state = np.array(
                [[-1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])
            self.high_state = np.array(
                [[1 for _ in range(self.look_back)] for _ in range(int(self.N_SPAWN_CARS + 1) * 2 + 1)])

        # self.observation_space = gym.spaces.Box(low=-self.low_state, high=self.high_state,
        #                                         dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.look_back, 16),
                                                dtype=np.float32)
        action_low = np.array([-1])
        action_high = np.array([1])
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        # [cn, ..., c1, c0, normalized yaw angle, normalized speed error] => ci: coefficients
        self.state = np.zeros_like(self.observation_space.sample())

        # instances
        self.ego = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.acceleration_ = 0
        self.eps_rew = 0

        self.side_window = 5  # times 2 to make adjacent window

        self.motionPlanner = None
        self.vehicleController = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

    def seed(self, seed=None):
        pass

    def enumerate_actors(self, actors_norm_s_d, side_window, ego_s, ego_d, leading_s, leading_d, following_s,
                         following_d, left_s, left_d, leftUp_s, leftUp_d, leftDown_s, leftDown_d, lleft_s, lleft_d,
                         lleftUp_s, lleftUp_d, lleftDown_s, lleftDown_d, right_s, right_d, rightUp_s, rightUp_d,
                         rightDown_s, rightDown_d, rright_s, rright_d, rrightUp_s, rrightUp_d, rrightDown_s,
                         rrightDown_d):

        norm_s = [0 for _ in range(self.N_SPAWN_CARS)]
        norm_d = [0 for _ in range(self.N_SPAWN_CARS)]
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State']
            norm_s[i] = (act_s - ego_s) / self.max_s
            norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
            others_s[i] = act_s
            others_d[i] = act_d
        actors_norm_s_d.append(norm_s)
        actors_norm_s_d.append(norm_d)

        # --------------------------------------------- ego lane -------------------------------------------------
        same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) < 1)[0]
        if len(same_lane_d_idx) == 0:
            leading_s.append(1)
            #    leading_d.append(0)
            following_s.append(-1)
        #    following_d.append(0)
        else:
            #    same_lane_d = np.array(others_d)[same_lane_d_idx]
            same_lane_s = np.array(others_s)[same_lane_d_idx]
            s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1)),
                                   axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
            leading_s.append(
                norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > 0][0])] if (any(sorted_s_idx[:, 1] > 0)) else 1)
            #    leading_d.append(
            #        norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > 0][0])] if (any(sorted_s_idx[:, 1] > 0)) else 0)
            following_s.append(
                norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < 0][-1])] if (any(sorted_s_idx[:, 1] < 0)) else -1)
        #    following_d.append(
        #        norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < 0][-1])] if (any(sorted_s_idx[:, 1] < 0)) else 0)

        # --------------------------------------------- left lane -------------------------------------------------
        left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -3) * ((np.array(others_d) - ego_d) > -4))[0]
        if len(left_lane_d_idx) == 0:
            left_s.append(0)
            #    left_d.append(-0.33)

            leftUp_s.append(0.004)
            #    leftUp_d.append(-0.33)

            leftDown_s.append(-0.004)
        #    leftDown_d.append(-0.33)

        else:
            #    left_lane_d = np.array(others_d)[left_lane_d_idx]
            left_lane_s = np.array(others_s)[left_lane_d_idx]
            s_idx = np.concatenate((np.array(left_lane_d_idx).reshape(-1, 1), (left_lane_s - ego_s).reshape(-1, 1)),
                                   axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
            left_s.append(norm_s[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
                any(abs(sorted_s_idx[:, 1]) < side_window)) else -1)
            #    left_d.append(norm_d[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
            #        any(abs(sorted_s_idx[:, 1]) < side_window)) else -0.33)

            leftUp_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 1)
            #    leftUp_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
            #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else -0.33)

            leftDown_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -1)
        #    leftDown_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
        #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -0.33)

        # ------------------------------------------- two left lane -----------------------------------------------
        lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -6.5) * ((np.array(others_d) - ego_d) > -7.5))[0]
        if len(lleft_lane_d_idx) == 0:
            lleft_s.append(0)
            #    lleft_d.append(-0.66)

            lleftUp_s.append(0.004)
            #    lleftUp_d.append(-0.66)

            lleftDown_s.append(-0.004)
        #    lleftDown_d.append(-0.66)

        else:
            #    lleft_lane_d = np.array(others_d)[lleft_lane_d_idx]
            lleft_lane_s = np.array(others_s)[lleft_lane_d_idx]
            s_idx = np.concatenate((np.array(lleft_lane_d_idx).reshape(-1, 1), (lleft_lane_s - ego_s).reshape(-1, 1)),
                                   axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
            lleft_s.append(norm_s[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
                any(abs(sorted_s_idx[:, 1]) < side_window)) else -1)
            #    lleft_d.append(norm_d[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
            #        any(abs(sorted_s_idx[:, 1]) < side_window)) else -0.66)

            lleftUp_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 1)
            #    lleftUp_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
            #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else -0.66)

            lleftDown_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -1)
        #    lleftDown_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
        #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -0.66)

        # ---------------------------------------------- rigth lane --------------------------------------------------
        right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 3) * ((np.array(others_d) - ego_d) < 4))[0]
        if len(right_lane_d_idx) == 0:
            right_s.append(0)
            #    right_d.append(0.33)

            rightUp_s.append(0.004)
            #    rightUp_d.append(0.33)

            rightDown_s.append(-0.004)
        #    rightDown_d.append(0.33)

        else:
            #    right_lane_d = np.array(others_d)[right_lane_d_idx]
            right_lane_s = np.array(others_s)[right_lane_d_idx]
            s_idx = np.concatenate((np.array(right_lane_d_idx).reshape(-1, 1), (right_lane_s - ego_s).reshape(-1, 1)),
                                   axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
            right_s.append(norm_s[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
                any(abs(sorted_s_idx[:, 1]) < side_window)) else -1)
            #    right_d.append(norm_d[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
            #        any(abs(sorted_s_idx[:, 1]) < side_window)) else 0.33)

            rightUp_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 1)
            #    rightUp_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
            #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 0.33)

            rightDown_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -1)
        #    rightDown_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
        #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else 0.33)

        # ------------------------------------------- two rigth lane --------------------------------------------------
        rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 6.5) * ((np.array(others_d) - ego_d) < 7.5))[0]
        if len(rright_lane_d_idx) == 0:
            rright_s.append(0)
            #    rright_d.append(0.66)

            rrightUp_s.append(0.004)
            #    rrightUp_d.append(0.66)

            rrightDown_s.append(-0.004)
        #    rrightDown_d.append(0.66)

        else:
            #    rright_lane_d = np.array(others_d)[rright_lane_d_idx]
            rright_lane_s = np.array(others_s)[rright_lane_d_idx]
            s_idx = np.concatenate((np.array(rright_lane_d_idx).reshape(-1, 1), (rright_lane_s - ego_s).reshape(-1, 1)),
                                   axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]
            rright_s.append(norm_s[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
                any(abs(sorted_s_idx[:, 1]) < side_window)) else -1)
            #    rright_d.append(norm_d[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < side_window][0])] if (
            #        any(abs(sorted_s_idx[:, 1]) < side_window)) else 0.66)

            rrightUp_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 1)
            #    rrightUp_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > side_window][0])] if (
            #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > side_window)) else 0.66)

            rrightDown_s.append(norm_s[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
                any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else -1)
        #    rrightDown_d.append(norm_d[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < side_window][-1])] if (
        #        any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < side_window)) else 0.66)

    def fix_representation(self, ego_norm_s, ego_norm_d, leading_s, leading_d, following_s, following_d, left_s,
                           left_d, leftUp_s, leftUp_d, leftDown_s, leftDown_d, lleft_s, lleft_d, lleftUp_s,
                           lleftUp_d, lleftDown_s, lleftDown_d, right_s, right_d, rightUp_s, rightUp_d, rightDown_s,
                           rightDown_d, rright_s, rright_d, rrightUp_s, rrightUp_d, rrightDown_s, rrightDown_d):

        ego_norm_s.extend(ego_norm_s[-1] for _ in range(self.look_back - len(ego_norm_s)))
        ego_norm_d.extend(ego_norm_d[-1] for _ in range(self.look_back - len(ego_norm_d)))
        leading_s.extend(leading_s[-1] for _ in range(self.look_back - len(leading_s)))
        # leading_d.extend(leading_d[-1] for _ in range(self.look_back - len(leading_d)))
        following_s.extend(following_s[-1] for _ in range(self.look_back - len(following_s)))
        # following_d.extend(following_d[-1] for _ in range(self.look_back - len(following_d)))
        left_s.extend(left_s[-1] for _ in range(self.look_back - len(left_s)))
        # left_d.extend(left_d[-1] for _ in range(self.look_back - len(left_d)))
        leftUp_s.extend(leftUp_s[-1] for _ in range(self.look_back - len(leftUp_s)))
        # leftUp_d.extend(leftUp_d[-1] for _ in range(self.look_back - len(leftUp_d)))
        leftDown_s.extend(leftDown_s[-1] for _ in range(self.look_back - len(leftDown_s)))
        # leftDown_d.extend(leftDown_d[-1] for _ in range(self.look_back - len(leftDown_d)))
        lleft_s.extend(lleft_s[-1] for _ in range(self.look_back - len(lleft_s)))
        # lleft_d.extend(lleft_d[-1] for _ in range(self.look_back - len(lleft_d)))
        lleftUp_s.extend(lleftUp_s[-1] for _ in range(self.look_back - len(lleftUp_s)))
        # lleftUp_d.extend(lleftUp_d[-1] for _ in range(self.look_back - len(lleftUp_d)))
        lleftDown_s.extend(lleftDown_s[-1] for _ in range(self.look_back - len(lleftDown_s)))
        # lleftDown_d.extend(lleftDown_d[-1] for _ in range(self.look_back - len(lleftDown_d)))
        right_s.extend(right_s[-1] for _ in range(self.look_back - len(right_s)))
        # right_d.extend(right_d[-1] for _ in range(self.look_back - len(right_d)))
        rightUp_s.extend(rightUp_s[-1] for _ in range(self.look_back - len(rightUp_s)))
        # rightUp_d.extend(rightUp_d[-1] for _ in range(self.look_back - len(rightUp_d)))
        rightDown_s.extend(rightDown_s[-1] for _ in range(self.look_back - len(rightDown_s)))
        # rightDown_d.extend(rightDown_d[-1] for _ in range(self.look_back - len(rightDown_d)))
        rright_s.extend(rright_s[-1] for _ in range(self.look_back - len(rright_s)))
        # rright_d.extend(rright_d[-1] for _ in range(self.look_back - len(rright_d)))
        rrightUp_s.extend(rrightUp_s[-1] for _ in range(self.look_back - len(rrightUp_s)))
        # rrightUp_d.extend(rrightUp_d[-1] for _ in range(self.look_back - len(rrightUp_d)))
        rrightDown_s.extend(rrightDown_s[-1] for _ in range(self.look_back - len(rrightDown_s)))
        # rrightDown_d.extend(rrightDown_d[-1] for _ in range(self.look_back - len(rrightDown_d)))

        lstm_obs = np.concatenate((np.array(ego_norm_d)[-self.look_back:], np.array(ego_norm_s)[-self.look_back:],
                                   np.array(leading_s)[-self.look_back:],
                                   np.array(following_s)[-self.look_back:],
                                   np.array(left_s)[-self.look_back:],
                                   np.array(leftUp_s)[-self.look_back:],
                                   np.array(leftDown_s)[-self.look_back:],
                                   np.array(lleft_s)[-self.look_back:],
                                   np.array(lleftUp_s)[-self.look_back:],
                                   np.array(lleftDown_s)[-self.look_back:],
                                   np.array(right_s)[-self.look_back:],
                                   np.array(rightUp_s)[-self.look_back:],
                                   np.array(rightDown_s)[-self.look_back:],
                                   np.array(rright_s)[-self.look_back:],
                                   np.array(rrightUp_s)[-self.look_back:],
                                   np.array(rrightDown_s)[-self.look_back:]),
                                  axis=0)

        return lstm_obs.reshape(self.observation_space.shape[1], -1).transpose()  # state

    def non_fix_representation(self, speeds, ego_norm_s, ego_norm_d, actors_norm_s_d):
        speeds.extend(0 for _ in range(self.look_back - len(speeds)))
        ego_norm_s.extend(0 for _ in range(self.look_back - len(ego_norm_s)))
        ego_norm_d.extend(0 for _ in range(self.look_back - len(ego_norm_d)))
        actors_norm_s_d.extend([0 for _ in range(self.N_SPAWN_CARS)]
                               for _ in range(self.look_back * 2 - len(actors_norm_s_d)))

        # LSTM input
        speeds_vec = (np.array(speeds) - self.maxSpeed) / self.maxSpeed
        actors_norm_s_d_flattened = np.concatenate(np.array(actors_norm_s_d), axis=0)
        lstm_obs = np.concatenate(
            (np.array(speeds_vec), np.array(ego_norm_s), np.array(ego_norm_d), actors_norm_s_d_flattened), axis=0)
        lstm_obs = lstm_obs.reshape((self.N_SPAWN_CARS + 1) * 2 + 1, -1)
        return lstm_obs[:, -self.look_back:]  # state

    def step(self, action=None):
        self.n_step += 1
        lanechange = False
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
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]
        fpath, lanechange = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action, Tf=5, Vf_n=-1)
        wps_to_go = len(fpath.t) - 3  # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1

        speeds = []
        accelerations = []
        # actors_norm_s = []    # relative frenet s value wrt ego
        # actors_norm_d = []    # relative frenet d value wrt ego
        actors_norm_s_d = []  # relative frenet consecutive s and d values wrt ego
        ego_norm_s = []
        ego_norm_d = []
        leading_s = []
        leading_d = []
        following_s = []
        following_d = []
        left_s = []
        left_d = []
        leftUp_s = []
        leftUp_d = []
        leftDown_s = []
        leftDown_d = []
        lleft_s = []
        lleft_d = []
        lleftUp_s = []
        lleftUp_d = []
        lleftDown_s = []
        lleftDown_d = []
        right_s = []
        right_d = []
        rightUp_s = []
        rightUp_d = []
        rightDown_s = []
        rightDown_d = []
        rright_s = []
        rright_d = []
        rrightUp_s = []
        rrightUp_d = []
        rrightDown_s = []
        rrightDown_d = []

        # side_window = 5  # times 2 to make adjacent window

        # dictionary={'ego...':egos, ...}

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()

        # follows path until end of WPs for max 1.8seconds or loop counter breaks unless there is a langechange
        loop_counter = 0
        while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T * 1.5 or
                                          loop_counter < self.loop_break or lanechange):

            loop_counter += 1
            # for _ in range(wps_to_go):
            # self.f_idx += 1
            ego_location = [self.ego.get_location().x, self.ego.get_location().y,
                            math.radians(self.ego.get_transform().rotation.yaw)]
            self.f_idx = closest_wp_idx(ego_location, fpath, self.f_idx)
            # cmdSpeed = math.sqrt((fpath.s_d[self.f_idx]) ** 2 + (fpath.d_d[self.f_idx]) ** 2)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # overwite command speed usnig IDM
            vehicle_ahead = self.ego_los_sensor.get_vehicle_ahead()
            cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)

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
                    self.world_module.points_to_draw['path wp {}'.format(i)] = [
                        carla.Location(x=fpath.x[i], y=fpath.y[i]),
                        'COLOR_ALUMINIUM_0']
                self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
                self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
                self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            speed_ = get_speed(self.ego)  # speed in previous tick
            self.module_manager.tick()  # Update carla world
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()

            speed = get_speed(self.ego)
            acc = (speed - speed_) / self.dt
            speeds.append(speed)
            accelerations.append(acc)
            ego_s, ego_d = fpath.s[self.f_idx], fpath.d[self.f_idx]
            ego_norm_s.append((ego_s - self.init_s) / self.track_length)
            ego_norm_d.append(round(ego_d / (2 * self.LANE_WIDTH),2))
            # lstm_state = np.zeros_like(self.observation_space.sample())
            self.enumerate_actors(actors_norm_s_d, self.side_window, ego_s, ego_d, leading_s, leading_d, following_s,
                                  following_d, left_s, left_d, leftUp_s, leftUp_d, leftDown_s, leftDown_d, lleft_s,
                                  lleft_d, lleftUp_s, lleftUp_d, lleftDown_s, lleftDown_d, right_s, right_d,
                                  rightUp_s, rightUp_d, rightDown_s, rightDown_d, rright_s, rright_d, rrightUp_s,
                                  rrightUp_d, rrightDown_s, rrightDown_d)


            # if ego off-the road or collided
            if any(collision_hist) or ego_d < -4.5 or ego_d > 8:
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True
                break
            # if loop_counter >= self.loop_break:
            #    break

        """
                *********************************************************************************************************************
                *********************************************** RL Observation ******************************************************
                *********************************************************************************************************************
        """
        meanSpeed = np.mean(speeds)
        meanAcc = np.mean(accelerations)
        speed_n = (meanSpeed - self.targetSpeed) / self.targetSpeed  # -1<= speed_n <=1
        acc_n = meanAcc / (2 * self.maxAcc)  # -1<= acc_n <=1

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.state = self.fix_representation(ego_norm_s, ego_norm_d, leading_s, leading_d, following_s,
                                                 following_d, left_s, left_d, leftUp_s, leftUp_d, leftDown_s,
                                                 leftDown_d, lleft_s, lleft_d, lleftUp_s, lleftUp_d, lleftDown_s,
                                                 lleftDown_d, right_s, right_d, rightUp_s, rightUp_d, rightDown_s,
                                                 rightDown_d, rright_s, rright_d, rrightUp_s, rrightUp_d, rrightDown_s,
                                                 rrightDown_d)

            # self.state = lstm_obs.reshape(self.observation_space.shape[0], -1)

            # print(3 * '---EPS UPDATE---')
            # print(TENSOR_ROW_NAMES[0].ljust(15),
            #       '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
            # for idx in range(2, self.state.shape[1]):
            #    print(TENSOR_ROW_NAMES[idx - 1].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))
            # self.state = lstm_obs[:, -self.look_back:]
        else:
            # pad the feature lists to recover from the cases where the length of path is less than look_back time
            self.state = self.non_fix_representation(speeds, ego_norm_s, ego_norm_d, actors_norm_s_d)
            # self.state = lstm_obs[:, -self.look_back:]

        # print(self.state)
        # print(100 * '--')
        """
                **********************************************************************************************************************
                ********************************************* RL Reward Function *****************************************************
                **********************************************************************************************************************
        """
        # w_acc = 1 / 2
        # r_acc = np.exp(-abs(meanAcc) ** 2 / (2 * self.maxAcc) * w_acc) - 1  # -1<= r_acc <= 0
        w_speed = 10
        e_speed = abs(self.targetSpeed - speed)
        r_speed = 5 * np.exp(-e_speed ** 2 / self.maxSpeed * w_speed)  # 0<= r_speed <= 1
        r_laneChange = -abs(np.round(action[0])) / 10  # -1<= r_laneChange <= 0
        positives = r_speed
        # negatives = (r_acc + r_laneChange) / 2
        negatives = r_laneChange
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
            # print('eps rew: ', self.n_step, self.eps_rew)
            return self.state, reward, done, {'reserved': 0}
        if track_finished:
            # print('Finished the race')
            reward = 10
            done = True
            self.eps_rew += reward
            # print('eps rew: ', self.n_step, self.eps_rew)
            return self.state, reward, done, {'reserved': 0}

        self.eps_rew += reward
        # print(self.n_step, self.eps_rew)
        # print(reward, action)
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

        # self.state = np.zeros_like(self.observation_space.sample())

        speeds = []
        actors_norm_s_d = []  # relative frenet consecutive s and d values wrt ego
        ego_norm_s = []
        ego_norm_d = []
        leading_s = []
        leading_d = []
        following_s = []
        following_d = []
        left_s = []
        left_d = []
        leftUp_s = []
        leftUp_d = []
        leftDown_s = []
        leftDown_d = []
        lleft_s = []
        lleft_d = []
        lleftUp_s = []
        lleftUp_d = []
        lleftDown_s = []
        lleftDown_d = []
        right_s = []
        right_d = []
        rightUp_s = []
        rightUp_d = []
        rightDown_s = []
        rightDown_d = []
        rright_s = []
        rright_d = []
        rrightUp_s = []
        rrightUp_d = []
        rrightDown_s = []
        rrightDown_d = []

        ego_norm_s.append(0)
        ego_norm_d.append(round(init_d / (2 * self.LANE_WIDTH),2))
        speeds.append(0)

        self.enumerate_actors(actors_norm_s_d, self.side_window, self.init_s, init_d, leading_s, leading_d, following_s,
                              following_d, left_s, left_d, leftUp_s, leftUp_d, leftDown_s, leftDown_d, lleft_s,
                              lleft_d, lleftUp_s, lleftUp_d, lleftDown_s, lleftDown_d, right_s, right_d,
                              rightUp_s, rightUp_d, rightDown_s, rightDown_d, rright_s, rright_d, rrightUp_s,
                              rrightUp_d, rrightDown_s, rrightDown_d)

        if cfg.GYM_ENV.FIXED_REPRESENTATION:
            self.state = self.fix_representation(ego_norm_s, ego_norm_d, leading_s, leading_d, following_s,
                                                 following_d, left_s, left_d, leftUp_s, leftUp_d, leftDown_s,
                                                 leftDown_d, lleft_s, lleft_d, lleftUp_s, lleftUp_d, lleftDown_s,
                                                 lleftDown_d, right_s, right_d, rightUp_s, rightUp_d, rightDown_s,
                                                 rightDown_d, rright_s, rright_d, rrightUp_s, rrightUp_d, rrightDown_s,
                                                 rrightDown_d)

            # print(3 * '---RESET---')
            # print(TENSOR_ROW_NAMES[0].ljust(15),
            #       '{:+8.6f}  {:+8.6f}'.format(self.state[-1][1], self.state[-1][0]))
            # for idx in range(2, self.state.shape[1]):
            #    print(TENSOR_ROW_NAMES[idx - 1].ljust(15), '{:+8.6f}'.format(self.state[-1][idx]))
            # self.state = lstm_obs[:, -self.look_back:]
        else:
            # pad the feature lists to recover from the cases where the length of path is less than look_back time

            self.state = self.non_fix_representation(speeds, ego_norm_s, ego_norm_d, actors_norm_s_d)

        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        # print(self.state)
        # return np.array(self.state)
        return self.state

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

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

        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.IDM = IntelligentDriverModel(self.ego)

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
