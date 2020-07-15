#------------------------------------------------------------------------------
# Name:        cannonball_shot_demo.py
# Purpose:     simulation and plot of a cannon shot
#
# Author:      mueller (based on Matlab code of Prof. Wolfgang Nowak
#
# Created:     31.01.2017
# Copyright:   (c) mueller 2017
# Licence:     <your licence>
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import scipy

import cannonball_shot

plt.ion()

# 1 simulation ################################################################
# the trajectory depends on...
# initial velocity      v0          (1000)
# initial angle of shot a0          (45)
# cannonball radius     r           (0.1)
# wind velocity         [wx, wy]    (-10,0)
#
# the time step size dt (0.1) controls the accuracy of the simulation

v0 = 1000  # initial velocity      v0          (1000)
a0 = 45    # initial angle of shot a0          (45)
r  = 0.1   # cannonball radius     r           (0.1)
wx = -10   # wind velocity         [wx, wy]    (-10,0)
wy = 0
dt = 0.1   # time step size        dt          (0.1)

distance, duration, data = cannonball_shot.calc_shot(v0, a0, r, wx, wy, dt)

# plot the trajectory
plt.plot(data['x'], data['y'])

# label the axis
plt.ylabel('Height [m]')
plt.xlabel('Range [m]')
plt.draw()

# n random simulations ########################################################
v0 = 1000  # initial velocity      v0          (1000)
a0 = 45    # initial angle of shot a0          (45)
r  = 0.1   # cannonball radius     r           (0.1)
# number of shots
n = 100
# wind velocity is now normally distributed with mean = -10 and std = 10
wx = scipy.random.normal(loc=-10, scale=10, size=n)
wy = 0
dt = 0.1   # time step size        dt          (0.1)

# calculate and plot each shot with random wind velocity individually
for ii in range(n):
    if (ii % 10) == 0:
        print ii

    iwx = wx[ii]
    # ... calculation
    distance1, duration1, data1 = cannonball_shot.calc_shot(v0, a0, r, iwx,
                                                            wy, dt)
    # ... plotting
    plt.plot(data1['x'], data1['y'], color = '0.8')
#    plt.draw()
    plt.pause(0.05)

# plot the shot with mean wind velocity
plt.plot(data['x'], data['y'], color = 'r')
plt.ylabel('Height [m]')
plt.xlabel('Range [m]')
