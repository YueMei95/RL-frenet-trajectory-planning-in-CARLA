"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from agents.local_planner import cubic_spline_planner

import time

# Parameter
MAX_SPEED = 150.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 4.0  # maximum acceleration [m/ss]  || Tesla model 3: 6.878
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
LANE_WIDTH = 3.5    # lane width [m]
LAT_CENTERS = np.arange(-1, 3)*LANE_WIDTH   # lateral centers
MAX_ROAD_WIDTH = 4.0  # maximum road width [m]
D_ROAD_W = 2.0  # road width sampling length [m]
DT = 0.1  # simulation time tick [s]
MAXT = 5.0  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
D_T = 1.0  # prediction timestep length (s)
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]
MAX_DIST_ERR = 4.0  # max distance error to update frenet states based on ego states

# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def update_frenet_coordinate(fpath, loc):
    """
    Finds best Frenet coordinates (s, d) in the path based on current position
    """

    min_e = float('inf')
    min_idx = -1
    for i in range(len(fpath.t)):
        e = euclidean_distance([fpath.x[i], fpath.y[i]], loc)
        if e < min_e:
            min_e = e
            min_idx = i

    if min_idx != len(fpath.t)-1:
        min_idx += 1

    s, s_d, s_dd = fpath.s[min_idx], fpath.s_d[min_idx], fpath.s_dd[min_idx]
    d, d_d, d_dd = fpath.d[min_idx], fpath.d_d[min_idx], fpath.d_dd[min_idx]

    return s, s_d, s_dd, d, d_d, d_dd


class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:

    def __init__(self):
        self.id = None
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

        self.v = []  # speed


def calc_frenet_paths(s, s_d, s_dd, d, d_d, d_dd):
    frenet_paths = []

    # generate path to each offset goal
    path_id = 0
    for di in LAT_CENTERS:

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT + D_T, D_T):
            fp = Frenet_path()
            lat_qp = quintic_polynomial(d, d_d, d_dd, di, 0.0, 0.0, Ti)

            for t in np.arange(0.0, Ti, DT):
                fp.t.append(t)
                fp.d.append(lat_qp.calc_point(t))
                fp.d_d.append(lat_qp.calc_first_derivative(t))
                fp.d_dd.append(lat_qp.calc_second_derivative(t))
                fp.d_ddd.append(lat_qp.calc_third_derivative(t))

            # Loongitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                tfp.id = path_id
                path_id += 1

                lon_qp = quartic_polynomial(s, s_d, s_dd, tv, 0.0, Ti)

                for t in tfp.t:
                    tfp.s.append(lon_qp.calc_point(t))
                    tfp.s_d.append(lon_qp.calc_first_derivative(t))
                    tfp.s_dd.append(lon_qp.calc_second_derivative(t))
                    tfp.s_ddd.append(lon_qp.calc_third_derivative(t))

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = KJ * Jp + KT * Ti + KD * (tfp.d[-1]) ** 2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)
    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.sqrt(dx ** 2 + dy ** 2))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    if len(ob) == 0:
        return True
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    okind = []
    for i in range(len(fplist)):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, f_state, ob):
    s, s_d, s_dd, d, d_d, d_dd = f_state
    fplist = calc_frenet_paths(s, s_d, s_dd, d, d_d, d_dd)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    mincost = float("inf")
    bestpath_idx = None
    for i, fp in enumerate(fplist):
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath_idx = i

    return bestpath_idx, fplist


# def generate_target_course(x, y):
#     csp = cubic_spline_planner.Spline2D(x, y)
#
#     s = np.arange(0, csp.s[-1], 0.1)
#     rx, ry, ryaw, rk = [], [], [], []
#     for i_s in s:
#         ix, iy = csp.calc_position(i_s)
#         rx.append(ix)
#         ry.append(iy)
#         ryaw.append(csp.calc_yaw(i_s))
#         rk.append(csp.calc_curvature(i_s))
#
#     return csp


class MotionPlanner:
    def __init__(self):
        self.path = None
        self.ob = []
        self.csp = None

    def update_global_route(self, global_route):
        wx = []
        wy = []
        for p in global_route:
            wx.append(p[0])
            wy.append(p[1])
        self.csp = cubic_spline_planner.Spline2D(wx, wy)

    def update_obstacles(self, ob):
        self.ob = ob

    def start(self, route):
        self.update_global_route(route)
        f_state = [0, 0, 0, 0, 0, 0]
        best_path_idx, fplist = frenet_optimal_planning(self.csp, f_state, self.ob)
        self.path = fplist[best_path_idx]

    def run_step(self, ego_state, idx):
        t0 = time.time()

        # Frenet state estimation [s, s_d, s_dd, d, d_d, d_dd]
        f_state = [self.path.s[idx], self.path.s_d[idx], self.path.s_dd[idx],
                   self.path.d[idx], self.path.d_d[idx], self.path.d_dd[idx]]

        # Update frenet state estimation when distance error gets large (option 2: re-initialize the planner)
        e = euclidean_distance(ego_state[0:2], [self.path.x[idx], self.path.y[idx]])
        if e > MAX_DIST_ERR:
            s, s_d, s_dd, d, d_d, d_dd = update_frenet_coordinate(self.path, ego_state[0:2])
            # f_state[0], f_state[3] = s, d
            f_state = s, s_d, s_dd, d, d_d, d_dd
        # f_state[1:3] = ego_state[2:]

        # Frenet motion planning
        best_path_idx, fplist = frenet_optimal_planning(self.csp, f_state, self.ob)
        self.path = fplist[best_path_idx]
        print(len(fplist))
        print('trajectory planning time: {} s'.format(time.time() - t0))
        return self.path, fplist
