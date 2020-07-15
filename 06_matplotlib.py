# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
from __future__ import unicode_literals
import timeit
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
START = timeit.default_timer() # to get the runtime of the program

#==============================================================================
print 'Illustrating the plot function...'
#==============================================================================

# start a new figure
plt.figure(figsize=(10, 7), dpi=150, tight_layout=True)

x_arr = np.arange(1, 100)
y_arr = np.log(x_arr)

plt.plot(x_arr, y_arr, color='black', linestyle='-', marker='x', markersize=2,
         alpha=0.7, label='Log')

plt.xlabel('arange')
plt.ylabel('Log arange')
plt.legend(framealpha=0.5)
plt.grid()
plt.title('plot example')
plt.savefig('plot.png')

#plt.show(block=False)

#==============================================================================
print '\n\nIllustrating the scatter function...'
#==============================================================================

plt.figure(figsize=(10, 7), dpi=150, tight_layout=True)

x_arr = np.random.random(20)
y_arr = np.random.random(x_arr.shape[0])
s_arr = np.random.random(x_arr.shape[0]) * 400
c_arr = np.random.random(x_arr.shape[0])

plt.scatter(x_arr, y_arr, s=s_arr, c=c_arr, alpha=0.9, marker='o',
            label='random')

plt.xlabel('random')
plt.ylabel('random')
plt.legend(framealpha=0.5)
plt.grid()
plt.title('scatter example')
plt.savefig('scatter.png')
#plt.show(block=False)


#==============================================================================
print '\n\nIllustrating the pcolormesh function...'
#==============================================================================
plt.figure(figsize=(10, 7), dpi=150, tight_layout=True)

z_m_arr = np.random.random(size=(20, 20))
x_arr = np.linspace(0, 10, z_m_arr.shape[0] + 1)

# how the coordinates affect the mesh
#x_arr = np.cumsum(np.linspace(0, 10, z_m_arr.shape[0] + 1))

y_arr = x_arr.copy()

x_m_arr, y_m_arr = np.meshgrid(x_arr, y_arr, indexing='ij')

# get colormap names from:
# http://matplotlib.org/examples/color/colormaps_reference.html

min_val = 0.2
max_val = 0.8

plt.pcolormesh(x_m_arr, y_m_arr, z_m_arr, vmin=min_val, vmax=max_val,
               alpha=0.8, cmap='gist_earth')

# plot without coordinates
#plt.pcolormesh(z_m_arr, vmin=min_val, vmax=max_val,
#               alpha=0.8, cmap='gist_earth')
plt.colorbar()
plt.title('pcolormesh example')
plt.savefig('pcolormesh.png')
#plt.show(block=False)


#==============================================================================
print '\n\nIllustrating the subplots function...'
#==============================================================================

x_arr = np.arange(0, 100, 1)
y_arr = np.cos(x_arr)
rand_arr = np.random.random(size=x_arr.shape[0]*100)

fig, axes = plt.subplots(2, 2)
#fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

fig.set_size_inches(10, 7)
fig.set_dpi(150)

axes[0, 0].plot(x_arr, y_arr, label='(0, 0)')
axes[0, 1].scatter(x_arr, y_arr, s=5, label='(0, 1)')
axes[1, 0].pcolormesh(np.random.random(size=(x_arr.shape[0], x_arr.shape[0])))
axes[1, 1].hist(rand_arr, bins=10, label='(1, 1)', rwidth=0.8, alpha=0.7)

for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        if i == 1 and j == 0:
            continue
        axes[i, j].legend(framealpha=0.7)

plt.subplots_adjust(wspace=0.3, hspace=0.2)

plt.suptitle(r'Title with TeX: $x_2, x^2, \mu, \beta, \gamma$\n\n')
plt.savefig('subplots.png')
#plt.show(block=False)


#==============================================================================
print '\n\nIllustrating the subplots function...'
#==============================================================================

plt.figure(figsize=(10, 7), dpi=150)
x_arr = np.arange(0, 100, 0.5)
rand_2d_arr = np.random.random(size=(x_arr.shape[0], x_arr.shape[0]))
y_arr = np.tan(x_arr)

sb_plt_shp = (2, 10)

long_top_axes = plt.subplot2grid(sb_plt_shp, (0, 0), rowspan=1, colspan=9)
cb_axes = plt.subplot2grid(sb_plt_shp, (0, 9), rowspan=1, colspan=1)
shrt_left_axes = plt.subplot2grid(sb_plt_shp, (1, 0), rowspan=1, colspan=5)
shrt_ryt_axes = plt.subplot2grid(sb_plt_shp, (1, 5), rowspan=1, colspan=5)

cb_map = long_top_axes.pcolormesh(rand_2d_arr, cmap='Blues')
long_top_axes.set_xlabel('rand_col')
long_top_axes.set_ylabel('rand_row')

shrt_left_axes.plot(x_arr, x_arr**2)
shrt_left_axes.set_xlabel('$x$')
shrt_left_axes.set_ylabel('$x^2$')
shrt_left_axes.grid()

shrt_ryt_axes.plot(x_arr, y_arr)
shrt_ryt_axes.set_xlabel('$x$')
shrt_ryt_axes.set_ylabel('$tan(x)$')
shrt_ryt_axes.grid()

plt.colorbar(cb_map, cb_axes)

plt.subplots_adjust(wspace=10, hspace=0.35, top=0.93)

plt.suptitle('Subplot2grid  Example')

plt.savefig('subplot2grid.png')

#plt.show(block=False)

STOP = timeit.default_timer()  # Ending time
print ('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START))

