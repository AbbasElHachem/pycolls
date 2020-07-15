'''
Created on Nov 22, 2017

@author: Faizan-Uni

path: P:/Synchronize/IWS/2016_DFG_SPATE/scripts_p3/test_7_plot_iters.py

'''

import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import shapefile as shp
import matplotlib.pyplot as plt
from descartes import PolygonPatch


plt.ioff()


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    in_pkl = main_dir / r'cp_classi_5_rules_opt_19800101_19891231_rev_04_cyth.pkl'
    in_ds_file = main_dir / r'NCAR_ds010.0_19800101_19891231_dailydata_europe.nc'
    backgrnd_shp_file = Path(r'P:\Synchronize\GIS\GIS Data of Countries\World Map.shp')
    out_figs_dir = main_dir / r'cp_classi_5_rules_opt_19800101_19891231_rev_04_cyth'

    fuzz_nos_arr = np.array([[0.0, 0.0, 0.4],
                             [-0.2, 0.2, 0.5],
                             [0.5, 0.8, 1.2],
                             [0.6, 1.0, 1.1]])  # the fifth one is dealt with later

    n_cp_val = 9999
    n_cps = 12

    os.chdir(main_dir)

#     out_figs_dir.

    if not out_figs_dir.exists():
        out_figs_dir.mkdir()

    in_ds = xr.open_dataset(in_ds_file)

    idx = pd.DatetimeIndex(in_ds.time.values).date
    print('First date:', idx[0])
    print('Last date:', idx[-1])
    print('Total time steps:', idx.shape[0])

    lons = in_ds.lon.values
    lats = in_ds.lat.values

    slps = in_ds.slp.values
    n_time_steps = slps.shape[0]
    slps_rav = slps.reshape(n_time_steps, -1)

    min_slp = np.min(slps_rav, axis=1)
    max_slp = np.max(slps_rav, axis=1)

    slp_anom = (slps_rav - min_slp[:, None]) / (max_slp - min_slp)[:, None]

    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    lons_mesh = np.where(lons_mesh > lons[-1], lons_mesh - 360, lons_mesh)

    n_pts = lons_mesh.shape[0] * lons_mesh.shape[1]

    with open(in_pkl, 'rb') as in_prms_hdl:
        out_prms_dict = pickle.load(in_prms_hdl)
        curr_obj_val = out_prms_dict['curr_obj_val']
        best_obj_val = out_prms_dict['best_obj_val']
        accept_iters = out_prms_dict['accept_iters']
        rand_acc_iters = out_prms_dict['rand_acc_iters']
        reject_iters = out_prms_dict['reject_iters']
        curr_anneal_temp = out_prms_dict['curr_anneal_temp']
        best_accept_iters = out_prms_dict['best_accept_iters']
        best_cps = out_prms_dict['best_cps']
        best_sel_cps = out_prms_dict['best_sel_cps']
        uni_cps = out_prms_dict['uni_cps']
        cp_rel_freqs = out_prms_dict['cp_rel_freqs']
        
    n_time_steps = best_sel_cps.shape[0]
    n_fuzz_nos = fuzz_nos_arr.shape[0]

#     reshape = lons_mesh.shape

    best_cps_mean_slp_anoms = np.empty(shape=(n_cps, n_pts),
                                      dtype=float)

    n_fuzz_no_arng = np.arange(0, n_fuzz_nos)
    n_cps_range = np.arange(0, n_cps)

    for j in range(n_cps):
        cp_idxs = best_sel_cps == j
        curr_slps = slp_anom[cp_idxs]
        curr_mean_slps = curr_slps.mean(axis=0)
        best_cps_mean_slp_anoms[j] = curr_mean_slps

    uni_cps, cps_freqs = np.unique(best_sel_cps, return_counts=True)
    cp_rel_freqs = 100 * cps_freqs / float(n_time_steps)
    cp_rel_freqs = np.round(cp_rel_freqs, 2)

    print('\n%-10s:%s' % ('Unique CPs', 'Relative Frequencies (%)'))
    for x, y in zip(uni_cps, cp_rel_freqs):
        print('%10d:%-20.2f' % (x, y))

    sf = shp.Reader(str(backgrnd_shp_file))
    poly_list = [i.__geo_interface__ for i in sf.iterShapes()]

    cont_levels = np.linspace(0.0, 1.0, 60)

    for j in range(best_cps.shape[0]):
        print('Plotting CP:', j)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.gca()
        curr_cp = best_cps_mean_slp_anoms[j]

        curr_false_idxs = best_cps[j] == n_fuzz_nos
        curr_cp[curr_false_idxs] = np.nan

        curr_cp = curr_cp.reshape(lons_mesh.shape)

        for poly in poly_list:
            ax.add_patch(PolygonPatch(poly,
                                      alpha=0.5,
                                      fc='#999999',
                                      ec='#999999'))

        cs = plt.contour(lons_mesh,
                         lats_mesh,
                         curr_cp,
                         levels=cont_levels,
                         vmin=0,
                         vmax=1.0,
                         linestyles='solid',
                         extend='both')

        plt.title('CP no. %d' % j)
        plt.xlim(lons_mesh.min(), lons_mesh.max())
        plt.ylim(lats_mesh.min(), lats_mesh.max())
#         plt.show(block=True)
        plt.clabel(cs, inline=True, inline_spacing=0.01, fontsize=10)

        plt.savefig(str(out_figs_dir / ('cp_map_nan_%0.2d.png' % j)))

        plt.close()
#         break

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
               ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))

