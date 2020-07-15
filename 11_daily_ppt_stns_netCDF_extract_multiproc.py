# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
import time
import timeit
import os
import osr
import numpy as np
from faizpy10 import list_full_path
import pandas as pd
import netCDF4 as nc
from multiprocessing import Process, Queue


def extract_grid_ppt(sub_netcdf_list, in_coords_df, sub_out_cat_data_df, ppt_que):

    '''
    Extract precipitation from a given list of netCDf files
        - The catchments shapefile can have one more catchment polygons.
        - Change names of variables and values inside the function,
          (not everything is specified through the arguments).
    '''


    xs = in_coords_df[['X']].loc[sub_out_cat_data_df.columns].values
    ys = in_coords_df[['Y']].loc[sub_out_cat_data_df.columns].values

    spt_ref = osr.SpatialReference()
    spt_ref.ImportFromEPSG(32718)
    trgt = osr.SpatialReference()
    trgt.ImportFromEPSG(4326)
    tfm = osr.CreateCoordinateTransformation(spt_ref, trgt)

    xs_tfm = []
    ys_tfm = []
    for xy  in zip(xs, ys):
        ppt_x, ppt_y = list(map(float, xy))
        ppt_tfm_pts = tfm.TransformPoint(ppt_x, ppt_y)
        #print 'Orig XY:', xy
        #print 'Trans XY:', ppt_tfm_pts

        xs_tfm.append(ppt_tfm_pts[0])
        ys_tfm.append(ppt_tfm_pts[1])

    for netcdf in sub_netcdf_list:
        #print 'Going through: %s' % netcdf
        in_nc = nc.Dataset(netcdf)
        lat_arr = in_nc.variables['latitude'][:]
        lon_arr = in_nc.variables['longitude'][:]

        # convert the netCDF time to regular time
        time_var = in_nc.variables['time']
        time_arr = nc.num2date(in_nc.variables['time'][:], time_var.units, calendar=time_var.calendar)

        ppt_var = in_nc.variables['precip']

        ppt_lat_idx_list = []
        ppt_lon_idx_list = []
        for xy in zip(xs_tfm, ys_tfm):
            lat_idx = np.argmin(np.abs(lat_arr - xy[1]))
            lon_idx = np.argmin(np.abs(lon_arr - xy[0]))

            ppt_lat_idx_list.append(lat_idx)
            ppt_lon_idx_list.append(lon_idx)

        for idx, date in enumerate(time_arr):
            if date in sub_out_cat_data_df.index:
                daily_ppt_grid = ppt_var[idx]
                stn_ppt_values = daily_ppt_grid[ppt_lat_idx_list, ppt_lon_idx_list]
                sub_out_cat_data_df.loc[date][sub_out_cat_data_df.columns] = stn_ppt_values

        #print '\n\n'
        in_nc.close()

    ppt_que.put(sub_out_cat_data_df)
    return None


if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer() # to get the runtime of the program

    #####
    # set values for the netCDf in the function above
    #####

    main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep/'
    coords_file = r'santa_ppt_coords.csv'
    out_cat_data_file = r'Peru_daily_ppt_stns_CHIRPS_time_ser.csv'
    netcdfs_dir = r'Q:\CHIRPS\daily'
    sep = ';'
    n_procs = 3

    out_start_date = '19810101'
    out_end_date = '20151231'

    test_ftn = False

    os.chdir(main_dir)

    all_netcdf_list = list_full_path('.nc', netcdfs_dir)

    in_coords_df = pd.read_csv(coords_file, sep=sep, index_col=0, encoding='utf-8')
    in_coords_df.dropna(inplace=True)

    out_date_range = pd.date_range(out_start_date, out_end_date)
    out_cat_data_df = pd.DataFrame(index=out_date_range, columns=in_coords_df.index)

    ppt_que = Queue()
    procs = []

    if isinstance(all_netcdf_list, list):
        #print 'is a list'
        pass
    else:
        #print 'not a list'
        all_netcdf_list = list([all_netcdf_list])

    n_procs = min(n_procs, len(all_netcdf_list))
    if test_ftn:
        extract_grid_ppt(all_netcdf_list, in_coords_df, out_cat_data_df, ppt_que)

    else:
        idxs = pd.np.linspace(0, len(all_netcdf_list), n_procs + 1, endpoint=True, dtype='int64')
        if idxs.shape[0] == 1:
            idxs = pd.np.concatenate((pd.np.array([0]), idxs))
            n_procs = 1

        idxs = np.unique(idxs)
        for proc_id in range(n_procs):
            sub_netcdf_list = all_netcdf_list[idxs[proc_id]:idxs[proc_id+1]]
            proc = Process(target=extract_grid_ppt, args=(sub_netcdf_list, in_coords_df, out_cat_data_df, ppt_que))
            procs.append(proc)
            proc.start()

        for proc in procs:
            sub_out_cat_data_df = ppt_que.get()
            out_cat_data_df.update(sub_out_cat_data_df)

        for proc in procs:
            proc.join()

        out_cat_data_df[out_cat_data_df < 0.1] = 0.0  # set values less than 0.1 to zero
        out_cat_data_df.to_csv(out_cat_data_file, sep=sep, float_format='%5.2f', encoding='utf-8')

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start))
