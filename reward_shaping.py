import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# d
max_d = 2
d = np.arange(0, max_d, 0.01)
r_d0 =  - d/max_d # 1 + np.sin(2 * np.pi * t)
# r_d1 = np.exp(-d**2/max_d)-1
w = 10
r_d2 = np.exp(-d**2/max_d*w)-1
fig, ax = plt.subplots()
ax.plot(d, r_d0)
# ax.plot(d, r_d1)
ax.plot(d, r_d2)
ax.set(xlabel='d', ylabel='rew',
       title='Reward functions')
ax.grid()
# plt.show()


# THETA
max_theta = math.pi/2
theta = np.arange(0, max_theta, 0.01)
r_theta0 =  - theta/max_theta # 1 + np.sin(2 * np.pi * t)
# r_theta1 = np.exp(-theta**2/max_theta)-1
w = 12
r_theta2 = np.exp(-theta**2/max_theta*w)-1
fig, ax2 = plt.subplots()
ax2.plot(theta, r_theta0)
# ax2.plot(theta, r_theta1)
ax2.plot(theta, r_theta2)
ax2.set(xlabel='theta', ylabel='rew',
       title='Reward functions')
ax2.grid()


# w
max_w = math.sqrt(2 * 180 ** 2) / 4
w = np.arange(0, max_w, 0.01)
r_w0 =  - w/max_w # 1 + np.sin(2 * np.pi * t)
# r_w1 = np.exp(-w**2/max_w)-1
ww = 1/5
r_w2 = np.exp(-w**2/max_w*ww)-1
fig, ax3 = plt.subplots()
ax3.plot(w, r_w0)
# ax3.plot(w, r_w1)
ax3.plot(w, r_w2)
ax3.set(xlabel='w', ylabel='rew',
       title='Reward functions')
ax3.grid()
plt.show()

