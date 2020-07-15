#-------------------------------------------------------------------------------
# Name:        cannonball_shot.py
# Purpose:     function to calculate the cannon shot
#
# Author:      mueller (based on Matlab code of Prof. Wolfgang Nowak
#
#
# Created:     31.01.2017
# Copyright:   (c) mueller 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np


def calc_shot(v0, a0, r, wx, wy, dt):
    # CALC_SHOT does three things:
    # (1) it simulates the trajectory of a cannonball
    # (2) it computes the shot distance and duration
    #
    # It uses a quadratic law for air friction
    # and considers wind against the shot direction
    # using this finite difference equation:
    #   dv = dt*( g - 1/2 * C_d * rho * A * (v-w).**2 / m)
    # where...
    #   dv  [m/s]    is a velocity change in time interval dt
    #   dt  [s]      is a discrete time for numerical solution of the ODE
    #   g   [m/s**2]  is the gravity acceleration vector
    #   C_d [-]      is the air friction coefficient of a sphere
    #   rho [kg/m**3] is the density of air
    #   A   [m**2]    is the cross section of the cannonball
    #   m   [kg]     is the mass of the cannonball
    #   v   [m/s]    is the velocity of the cannonball
    #   w   [m/s]    is the wind velocity
    #
    # subject to the initial conditions
    #   v(t=0) = v0  initial velocity
    #   x(t=0) = x0  initial x-position
    #   y(t=0) = y0  initial y-position (vertical coordinate)
    #   a(t=0) = a0  initial angle of cannonball flight
    #
    # the shot distance is the distance in x covered by the cannonball
    # when it touches the ground surface (y=0). The final time step is
    # modifies in such a manner that the cannonball exactly hits the ground
    # (and does not numerically go below the ground surface).


    # set initial conditions for cannonball
    t0   = 0                      # starting time of shot [s]
    x0   = 0                      # initial position of cannonball [m]
    y0   = 0                      # initial position of cannonball [m]
    v0   = v0                     # initial shot velocity [m/s]
    a0   = a0                     # initial angle of shot [degrees]

    # set problem parameters for cannonball flight
    g   = 9.81                       # gravitational constant [m**2/s]
    rho = 1.225                      # density of air [kg/m**3]
    C_d = 0.5                        # friction coefficient of cannonball [-]
    r   = r                          # radius of cannonball [m]
    A   = np.pi * r**2              # cross-sectional area of cannonball
    m   = 8000. * 4/3 * np.pi * r**3 # mass of cannonball (density of steel = 8000 kg/m**3)
    wx  = wx                         # wind velocity [m/s] (negative is against shot direction)
    wy  = wy                         # wind velocity [m/s] (negative is against shot direction)

    # discretization for solving the cannonball ODE
    dt  = dt                         # time step size for dinite difference


    # initialize variables that change dynamically
    # during the finite difference simulation:
    t  = [t0]           # initial time
    x  = [x0]           # initial x position
    y  = [y0]           # initial y position
    a  = a0/360.*2*np.pi  # initial angle

    # decompose absolute velocity and angle into velocity components vx, vy:
    vx  = v0*np.cos(a)   # initial x-velocity
    vy  = v0*np.sin(a)   # initial y-velocity

    # let the ball fly until it hits the ground: loop!
    while 1==1:
        # use current velocities to update position
        x_new = x[-1] + vx*dt
        y_new = y[-1] + vy*dt
        t_new = t[-1] + dt

        # append the new position to the history of positions
        # (this is dirty code as it enlarges variables on-the-fly)
        # (but it is a simple solution to output the entire trajectory)
        x.append(x_new)
        y.append(y_new)
        t.append(t_new)

        # check termination criterion: did the ball hit the ground?
        if y_new < -85:
            # if yes: ensure that last time step is so small
            # that cannonball is not BELOW ground durface!
            dt_back = -y[-1]/vy           # (this is how much to go back in time)
            x[-1] = x[-1] + vx*dt_back
            y[-1] = y[-1] + vy*dt_back
            t[-1] = t[-1] + dt_back
            break # finished: leave the shot simulation


        # if cannonball is still flying: update velocities (here: gravity)
        vy = vy - g*dt

        # update velocities (here: air friction with wind)
        vx_effective = vx - wx
        vy_effective = vy - wy
        v_effective = np.sqrt(vx_effective**2 + vy_effective**2)
        ax = - 1/2 * C_d * rho * A * vx_effective * v_effective /m
        ay = - 1/2 * C_d * rho * A * vy_effective * v_effective /m

        vx = vx + ax * dt
        vy = vy + ay * dt


    # compile output (x,y,t) into DATA dictionairy
    data = {}
    data['x'] = x
    data['y'] = y
    data['t'] = t

    # return shot distance as last x position
    distance = x[-1]
    duration = t[-1]

    return distance, duration, data

