# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
from __future__ import unicode_literals
import time
import timeit
import os
from multiprocessing import Process, Queue

import ogr
import osr
import numpy as np
import pandas as pd
import netCDF4 as nc

ogr.UseExceptions()
osr.UseExceptions()


def extract_grid_ppt(sub_date_list,
                     in_net_cdf,
                     in_cat_shp,
                     sub_out_cat_data_df,
                     other_args,
                     ppt_que):

    '''
    Extract precipitation from a given list of netCDf files
        The catchments shapefile can have one more catchment polygons.
        Change names of variables and values inside the function,
        (not everything is specified through the arguments).
    '''
    cat_vec = ogr.Open(in_cat_shp)
    lyr = cat_vec.GetLayer(0)

    spt_ref = lyr.GetSpatialRef()
    trgt = osr.SpatialReference()
    trgt.ImportFromEPSG(other_args[1])
    tfm = osr.CreateCoordinateTransformation(spt_ref, trgt)
    back_tfm = osr.CreateCoordinateTransformation(trgt, spt_ref)

    #raise Exception

    feat_dict = {}
    feat_area_dict = {}
    cat_area_ratios_dict = {}
    cat_envel_dict = {}

    feat = lyr.GetNextFeature()

    while feat:
        geom = feat.GetGeometryRef()
        f_val = feat.GetFieldAsString(str(other_args[0]))
        if f_val is None:
            raise RuntimeError

        feat_area_dict[f_val] = geom.Area() # do before transform

        geom.Transform(tfm)
        feat_dict[f_val] = feat

        cat_envel_dict[f_val] = geom.GetEnvelope() # do after transform
        feat = lyr.GetNextFeature()

    #print 'Going through: %s' % netcdf
    in_nc = nc.Dataset(in_net_cdf)
    lat_arr = in_nc.variables[other_args[4]][:]
    lon_arr = in_nc.variables[other_args[3]][:]

    apply_cell_correc = other_args[6]

    # convert the netCDF time to regular time
    time_var = in_nc.variables[other_args[2]]
    time_arr =  nc.num2date(in_nc.variables[other_args[2]][:],
                            time_var.units,
                            calendar=time_var.calendar)

    ppt_var = in_nc.variables[other_args[5]]

    #print 'Counting time from (in the netCDF file):',time_var.units
    #print 'Start date in the netCDF: ', time_arr[0]
    #print 'End date in the netCDF: ', time_arr[-1]
    #print 'Total time steps in the netCDF: ', time_arr.shape[0]

    cell_size = round(lon_arr[1] - lon_arr[0], 3)
    x_l_c = lon_arr[0]
    x_u_c = lon_arr[-1]
    y_l_c = lat_arr[0]
    y_u_c = lat_arr[-1]

    flip_lr = False
    flip_ud = False

    if x_l_c > x_u_c:
        x_l_c, x_u_c = x_u_c, x_l_c
        flip_lr = True

    if y_l_c > y_u_c:
        y_l_c, y_u_c = y_u_c, y_l_c
        flip_ud = True

    #raise Exception

    if apply_cell_correc:  # because CHIRPS has values at the center of the cell
                               # so I shift it back
        x_l_c -= (cell_size/2.)
        x_u_c -= (cell_size/2.)
        y_l_c -= (cell_size/2.)
        y_u_c -= (cell_size/2.)

    x_coords = np.arange(x_l_c, x_u_c * 1.00000001, cell_size)
    y_coords = np.arange(y_l_c, y_u_c * 1.00000001, cell_size)

    cat_x_idxs_dict = {}
    cat_y_idxs_dict = {}

#    print feat_dict.keys()

    for cat_no in feat_dict.keys():
        #print 'Cat no:', cat_no
        geom = feat_dict[cat_no].GetGeometryRef()

        extents = cat_envel_dict[cat_no]
        cat_area = feat_area_dict[cat_no]

        inter_areas = []
        x_low, x_hi, y_low, y_hi = extents

        # adjustment to get all cells intersecting the polygon
        x_low = x_low - cell_size
        x_hi = x_hi + cell_size
        y_low = y_low - cell_size
        y_hi = y_hi + cell_size

        x_cors_idxs = np.where(np.logical_and(x_coords >= x_low,
                                              x_coords <= x_hi))[0]
        y_cors_idxs = np.where(np.logical_and(y_coords >= y_low,
                                              y_coords <= y_hi))[0]

        x_cors = x_coords[x_cors_idxs]
        y_cors = y_coords[y_cors_idxs]

        cat_x_idxs = []
        cat_y_idxs = []

        for x_idx in range(x_cors.shape[0] - 1):
            for y_idx in range(y_cors.shape[0] - 1):
                ring = ogr.Geometry(ogr.wkbLinearRing)

                ring.AddPoint(x_cors[x_idx], y_cors[y_idx])
                ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx])
                ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx + 1])
                ring.AddPoint(x_cors[x_idx], y_cors[y_idx + 1])
                ring.AddPoint(x_cors[x_idx], y_cors[y_idx])

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                inter_poly = poly.Intersection(geom)

                # to get the area, I convert it to coordinate sys of
                # the shapefile that is hopefully in linear units
                inter_poly.Transform(back_tfm)
                inter_area = inter_poly.Area()

                inter_areas.append(inter_area)

                cat_x_idxs.append((x_cors[x_idx] - x_l_c) / cell_size)
                cat_y_idxs.append((y_cors[y_idx] - y_l_c) / cell_size)

        cat_area_ratios_dict[cat_no] = np.divide(inter_areas, cat_area)

        cat_x_idxs_dict[cat_no] = np.int64(np.round(cat_x_idxs, 6))
        cat_y_idxs_dict[cat_no] = np.int64(np.round(cat_y_idxs, 6))
#        print 'Normalized area sum:', np.sum(cat_area_ratios_dict[cat_no])
#
#    for cat_no in feat_dict.keys():
#        print cat_no
#        print cat_y_idxs_dict[cat_no]
#        print cat_x_idxs_dict[cat_no]
#        print '\n'

    for idx, date in enumerate(time_arr):
        if date in sub_out_cat_data_df.index:
            all_ppt_vals = ppt_var[idx]

            if flip_lr:
                all_ppt_vals = np.fliplr(all_ppt_vals)
            if flip_ud:
                all_ppt_vals = np.flipud(all_ppt_vals)

            for cat_no in feat_dict.keys():
                ppt_vals = all_ppt_vals[cat_y_idxs_dict[cat_no],
                                        cat_x_idxs_dict[cat_no]]
                fin_ppt_vals = np.multiply(ppt_vals,
                                           cat_area_ratios_dict[cat_no])
                sub_out_cat_data_df.loc[date][cat_no] = \
                    round(np.sum(fin_ppt_vals), 2)

    in_nc.close()
    ppt_que.put(sub_out_cat_data_df)
    cat_vec.Destroy()
    return


if __name__ == '__main__':

    print ('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer() # to get the runtime of the program

    #####
    # set values for the netCDf in the function above
    #####

    main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep'

    # a list of some arguments
    # [field name to use as catchment names / numbers,
    # netCDF EPSG code,
    # netCDF time name,
    # netCDF X coords name,
    # netCDF Y coords name,
    # netCDF variable to read,
    # apply cell corner correction]
    other_args = ['DN', 32718, 'time', 'X', 'Y', 'EDK', True]

#==============================================================================
#    # TEMPERATURE
#    in_net_cdf = os.path.join(
#            r'santa_temperature_kriging_01',
#            r'santa_temp_kriging_1963-01-01_to_2015-12-31_5km.nc')
#    out_cat_data_file = os.path.join(
#            r'santa_temperature_kriging_01',
#            r'santa_monthly_temperature_%s.csv' % other_args[5])
#==============================================================================

#==============================================================================
#    # PRECIPITATION
#    in_net_cdf = os.path.join(
#            r'santa_precipitation_kriging_01',
#            r'santa_ppt_kriging_1963-01-01_to_2015-12-31_5km.nc')
#    out_cat_data_file = os.path.join(
#            r'santa_precipitation_kriging_01',
#            r'santa_monthly_precipitation_%s.csv' % other_args[5])
#==============================================================================

#==============================================================================
    # POTENTIAL EVAPOTRANSPIRATION
    in_net_cdf = os.path.join(
            r'santa_pet_kriging_01',
            r'santa_pet_kriging_1963-01-01_to_2015-12-31_5km.nc')
    out_cat_data_file = os.path.join(
            r'santa_pet_kriging_01',
            r'santa_monthly_pet_%s.csv' % other_args[5])
#==============================================================================

    in_cat_shp = r'santa_rs_minerve_prep_may17\taudem_out\watersheds.shp'

    sep = str(';')
    n_procs = 3

    out_start_date = '19630101'
    out_end_date = '20151231'
    freq = 'M'

    test_ftn = False

    os.chdir(main_dir)

    cat_vec = ogr.Open(in_cat_shp)
    lyr = cat_vec.GetLayer(0)
    #raise Exception

    feat_area_dict = {}

    for feat in lyr:  # just to get the names of the catchments
        geom = feat.GetGeometryRef()
        f_val = feat.GetFieldAsString(str(other_args[0]))
        if f_val is None:
            raise RuntimeError
        feat_area_dict[f_val] = geom.Area()
    #raise Exception

    out_date_range = pd.date_range(out_start_date, out_end_date, freq=freq)
    out_cat_data_df = pd.DataFrame(index=out_date_range,
                                   columns=feat_area_dict.keys(),
                                   dtype=float)

    cat_vec.Destroy()

    ppt_que = Queue()
    procs = []

    n_procs = min(n_procs, out_date_range.shape[0])
    if test_ftn:
        extract_grid_ppt(out_date_range,
                         in_net_cdf,
                         os.path.join(main_dir, in_cat_shp),
                         out_cat_data_df,
                         other_args,
                         ppt_que)

    else:
        idxs = pd.np.linspace(0,
                              out_date_range.shape[0],
                              n_procs + 1,
                              endpoint=True,
                              dtype='int64')
        if idxs.shape[0] == 1:
            idxs = pd.np.concatenate((pd.np.array([0]), idxs))
            n_procs = 1

        idxs = np.unique(idxs)
        for proc_id in range(n_procs):
            sub_date_list =  out_date_range[idxs[proc_id]:idxs[proc_id+1]]
            proc = Process(target=extract_grid_ppt,
                           args=(sub_date_list,
                                 in_net_cdf,
                                 os.path.join(main_dir, in_cat_shp),
                                 out_cat_data_df,
                                 other_args,
                                 ppt_que))
            procs.append(proc)
            proc.start()

        for proc in procs:
            sub_out_cat_data_df = ppt_que.get()
            out_cat_data_df.update(sub_out_cat_data_df)

        for proc in procs:
            proc.join()

        out_cat_data_df.to_csv(out_cat_data_file, sep=sep, float_format='%.2f')

    stop = timeit.default_timer()  # Ending time
    print '\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start)
