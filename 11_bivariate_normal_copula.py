# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
import timeit
import time
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, pi
from scipy.stats import norm
import matplotlib.cm as cmaps
plt.ioff()

print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
start = timeit.default_timer() # to get the runtime of the program

bins = 200
rho = 0.8
m = 3.0
k = 1.0
alpha = 1.0

u1 = np.linspace(1./bins, 1 - 1./bins, bins)
u2 = u1.copy()

t1 = norm.ppf(u1)
t2 = t1.copy()

t1_v = np.where(t1 >= m, k * ((t1 - m)**alpha), m - t1)
t1_v = t1_v[np.argsort(t1_v)]
t2_v = t1_v.copy()


def bivar_gau_dens(t1, t2, rho):
    dens = exp(-0.5 * (t1**2 + t2**2 - 2*rho*t1*t2) / (1 - rho**2))
    dens /= 2.0 * pi * sqrt(1 - rho**2)
    return dens


def bivar_gau_cop(t1, t2, rho):
    cop_dens = exp(-0.5 * (rho / (1 - rho**2)) * ((rho*(t1**2 + t2**2)) - 2*t1*t2))
    cop_dens /= sqrt(1 - rho**2)
    return cop_dens


def univar_gau_dens_v(t, m, k, alpha):
    dens = 0.0
    dens += (1. / (k * alpha)) * ((t / k)**((1./alpha) - 1.)) * norm.pdf(((t / k)**(1./alpha)) + m)
    dens += norm.pdf(-t + m)
    return dens


def bivar_gau_dens_v(t1, t2, rho, m, k, alpha):
    dens = 0.0
    dens += (1./((k**2) * (alpha**2))) * (((t1 * t2) / k**2)**((1./alpha) - 1.)) * bivar_gau_dens(((t1/k)**(1./alpha)) + m, ((t2/k)**(1./alpha)) + m, rho)
    dens += (1./(k * alpha)) * ((t1 / k)**((1./alpha) - 1)) * bivar_gau_dens(((t1/k)**(1./alpha)) + m, -t2 + m, rho)
    dens += (1./(k * alpha)) * ((t2 / k)**((1./alpha) - 1)) * bivar_gau_dens(-t1 + m, ((t2/k)**(1./alpha)) + m, rho)
    dens += bivar_gau_dens(-t1 + m, -t2 + m, rho)
    return dens


def bivar_gau_cop_v(t1, t2, rho, m, k, alpha):
    cop_dens = bivar_gau_dens_v(t1, t2, rho, m, k, alpha)
    cop_dens /= univar_gau_dens_v(t1, m, k, alpha)
    cop_dens /= univar_gau_dens_v(t2, m, k, alpha)
    return cop_dens

dudv = (1. / bins)**2
area_n = 0.0
area_v = 0.0

my_copula = np.zeros(shape=(bins, bins))
my_copula_v = np.zeros(shape=(bins, bins))
for i in range(bins):
    for j in range(bins):
        my_copula[i, j] = bivar_gau_cop(t1[i], t2[j], rho)
        my_copula_v[i, j] = bivar_gau_cop_v(t1_v[i], t2_v[j], rho, m, k, alpha)
        area_n += my_copula[i, j] * dudv
        area_v += my_copula_v[i, j] * dudv

print area_n
print area_v

max_dens_val_ratio = 0.2
max_norm_cop = max_dens_val_ratio * my_copula.max()
max_norm_cop_v = max_dens_val_ratio * my_copula_v.max()

#my_copula[my_copula > max_norm_cop] = max_norm_cop
#my_copula_v[my_copula_v > max_norm_cop_v] = max_norm_cop_v

my_copula = my_copula / max_norm_cop
my_copula_v = my_copula_v / max_norm_cop_v

mesh_x, mesh_y = np.mgrid[0:bins, 0:bins] + 0.5
cop_ax_ticks = np.linspace(0, bins, 6, dtype='int')
cop_ax_labs = np.round(np.linspace(u1[0], u1[-1], 6, dtype='float'), 1)
title_font_size = 12

f, axes = plt.subplots(2, 3, figsize=(15, 9))
axes[0, 0].set_title('Normal Copula\n', fontsize=title_font_size)
p_ax = axes[0, 0].pcolormesh(mesh_x, mesh_y, my_copula, vmin=0, vmax=1.0, cmap=cmaps.gist_ncar)
axes[0, 0].set_xticks(cop_ax_ticks)
axes[0, 0].set_yticks(cop_ax_ticks)
axes[0, 0].set_xticklabels(cop_ax_labs)
axes[0, 0].set_yticklabels(cop_ax_labs)

axes[1, 0].set_title('t1\n(normal)\n', fontsize=title_font_size)
axes[1, 0].hist(t1, bins=20, alpha=0.5)
axes[1, 0].set_xlabel('t1')
axes[1, 0].set_ylabel('frequency')
axes[1, 0].grid()

axes[0, 1].set_title('V-Normal Copula\n', fontsize=title_font_size)
axes[0, 1].pcolormesh(mesh_x, mesh_y, my_copula_v, vmin=0, vmax=1.0, cmap=cmaps.gist_ncar)
axes[0, 1].set_xticks(cop_ax_ticks)
axes[0, 1].set_yticks(cop_ax_ticks)
axes[0, 1].set_xticklabels(cop_ax_labs)
axes[0, 1].set_yticklabels(cop_ax_labs)

axes[1, 1].set_title('t1\n(V-transformed)\n', fontsize=title_font_size)
axes[1, 1].hist(t1_v, bins=20, alpha=0.5)
axes[1, 1].set_xlabel('t1')
axes[1, 1].set_ylabel('frequency')
axes[1, 1].grid()

ax_l = axes[0, 2]
ax_l.set_axis_off()
cb = plt.colorbar(p_ax, ax=ax_l, fraction=1, aspect=10)
cb.set_label('copula density', size=10)
cb.set_ticks(np.linspace(0, 1.0, 11))

my_range = np.linspace(-3., 3., bins)
my_v_range = np.where(my_range >= m, k * (my_range - m)**alpha, m - my_range)
axes[1, 2].set_title('V-transformation\n', fontsize=title_font_size)
axes[1, 2].plot(my_range, my_v_range)
axes[1, 2].set_xlabel('t1')
axes[1, 2].set_ylabel('t1\n(V-transformed)')
axes[1, 2].grid()

plt.suptitle('Normal and V-Normal Copula\n(rho=%0.2f, m=%0.2f, k=%0.2f, alpha=%0.2f)\n' % (rho, m, k, alpha), fontsize=(title_font_size + 2))

#plt.show(block=False)
plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.85)
fig_loc = r'v_trans_norm_cop__rho_%0.2f__m_%0.2f__k_%0.2f__alpha_%0.2f.png' % (rho, m, k, alpha)
plt.savefig(fig_loc, bbox_inches='tight', dpi=300)

stop = timeit.default_timer()  # Ending time
print '\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start)

