# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

test simple kriging cython class
"""

import pyximport
import numpy as np
#pyximport.install()
pyximport.install(setup_args={"include_dirs":np.get_include()})
#                              "script_args":["--compiler=mingw64"]})
import KrigingCython as kriging
import timeit
import time

#xi = np.array([1., -1., -1., 2., 1.1, ])
#yi = np.array([0.5, -1., 1., 2., -1.1, ])
#zi = np.array([7, 5, 2.5, 7, 1.23])

#
#xk = np.array([0.])
#yk = np.array([0.])

#
#model = '0.9 Exp(10)'

#xi = np.array([-1., 1., 0., -1., 0.8, ])
#yi = np.array([-1., -1., 2., 2., 0.5, ])
#zi = np.array([4, 20, 14.2, 50, 10])

#a = 1
#b = 1.5
xi = np.array([0., 5., 0., 5., 5])
yi = np.array([0., 5., 5., 0., 0])
zi = np.array([53., 62., 52., 12., 35])
#zi = np.array([1., 1., 1., 1., 1.])
#si = a + b * zi
#si = np.array([0., 1., 0., 1., 0.])
si = np.array([1., 1., 1.5, 1., 1.])


xk = np.array([2.5, ])
yk = np.array([2.5, ])
#sk = np.array([a + b * 2.5, a + b * 5.0])
sk = np.array([1])

#model = '0.1 Nug(0.0) + 0.9 Sph(10)'
model = '1.0 Nug(0) + 0.9 Sph(50)'

#lim = 60.  # for indicator kriging

n = 1

ordinary = False
simple = False
external = False
ordinary_indicator = False
simple_indicator = False

#ordinary = True
#simple = True
external = True
#ordinary_indicator = True
#simple_indicator = True

print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
start = timeit.default_timer() # to get the runtime of the program

#==============================================================================
# # ordinary kriging
#==============================================================================
if ordinary:
    print '\a Ordinary Kriging...'
    for i in xrange(n):
        ordinary_kriging = kriging.OrdinaryKriging(xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
        ordinary_kriging.krige()

    print '\nDistances are:\n', ordinary_kriging.in_dists
    print '\nVariances are:\n', ordinary_kriging.in_vars
    print '\nRight hand sides are:\n', ordinary_kriging.rhss
    print '\nzks are:', ordinary_kriging.zk
    print '\nest_vars are:\n', ordinary_kriging.est_vars
    print '\nlambdas are:\n', ordinary_kriging.lambdas
    print '\nmus are:\n', ordinary_kriging.mus
    print '\n\n'

#==============================================================================
# # simple kriging
#==============================================================================
if simple:
    print '\a Simple Kriging...'
    for i in xrange(n):
        simple_kriging = kriging.SimpleKriging(xi=xi, yi=yi, zi=zi, xk=xk, yk=yk, model=model)
        simple_kriging.krige()

    print '\nDistances are:\n', simple_kriging.in_dists
    print '\nVariances are:\n', simple_kriging.in_covars
    print '\nRight hand sides are:\n', simple_kriging.rhss
    print '\nzks are:', simple_kriging.zk
    print '\nest_vars are:\n', simple_kriging.est_covars
    print '\nlambdas are:\n', simple_kriging.lambdas
    print '\n\n'

#==============================================================================
# # external drift kriging
#==============================================================================
if external:
    print '\a External Drift Kriging...'
    for i in xrange(n):
        external_drift_kriging = kriging.ExternalDriftKriging(xi=xi,
                                                                yi=yi,
                                                                zi=zi,
                                                                si=si,
                                                                xk=xk,
                                                                yk=yk,
                                                                sk=sk,
                                                                model=model)
        external_drift_kriging.krige()

    print '\nDistances are:\n', external_drift_kriging.in_dists
    print '\nVariances are:\n', external_drift_kriging.in_vars
    print '\nRight hand sides are:\n', external_drift_kriging.rhss
    print '\nzks are:\n', external_drift_kriging.zk
    print '\nlambdas are:\n', external_drift_kriging.lambdas
    print '\nmus_1 are:\n', external_drift_kriging.mus_1
    print '\nmus_2 are:\n', external_drift_kriging.mus_2
    print '\n\n'

#==============================================================================
# # indicator kriging
#==============================================================================
if ordinary_indicator:
    print '\a Indicator kriging based on ordinary kriging...'
    for i in xrange(n):
        oindicator_kriging = kriging.OrdinaryIndicatorKriging(xi=xi,
                                                                yi=yi,
                                                                zi=zi,
                                                                xk=xk,
                                                                yk=yk,
                                                                lim=lim,
                                                                model=model)
        oindicator_kriging.ikrige()

    print '\nDistances are:\n', oindicator_kriging.in_dists
    print '\nVariances are:\n', oindicator_kriging.in_vars
    print '\nixis are:\n', oindicator_kriging.ixi
    print '\niks are:\n', oindicator_kriging.ik
    print '\nest_vars are:\n', oindicator_kriging.est_vars
    print '\nlambdas are:\n', oindicator_kriging.lambdas
    print '\n\n'

if simple_indicator:
    print '\a Indicator kriging based on simple kriging...'
    for i in xrange(n):
        sindicator_kriging = kriging.SimpleIndicatorKriging(xi=xi,
                                                                yi=yi,
                                                                zi=zi,
                                                                xk=xk,
                                                                yk=yk,
                                                                lim=lim,
                                                                model=model)
        sindicator_kriging.ikrige()

    print '\nDistances are:\n', sindicator_kriging.in_dists
    print '\nVariances are:\n', sindicator_kriging.in_covars
    print '\nixis are:\n', sindicator_kriging.ixi
    print '\niks are:\n', sindicator_kriging.ik
    print '\nest_vars are:\n', sindicator_kriging.est_covars
    print '\nlambdas are:\n', sindicator_kriging.lambdas
    print '\n\n'


stop = timeit.default_timer()  # Ending time
print '\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start)

