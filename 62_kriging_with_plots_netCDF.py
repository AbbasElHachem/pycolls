# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
from __future__ import unicode_literals
import os
import timeit
import time

import numpy as np
import pandas as pd
import netCDF4 as nc
import gdal
import ogr
import shapefile as shp

import matplotlib.pyplot as plt

import pyximport
pyximport.install()

from KrigingCython import (OrdinaryKriging,
                           SimpleKriging,
                           ExternalDriftKriging)

plt.ioff()


print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
START = timeit.default_timer() # to get the runtime of the program

main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep'

#==============================================================================
## TEMPERATURE
#in_data_file = os.path.join(
#        r'santa_infill_temperature_Fabian_May2017_07_monthly_resamp',
#        r'infilled_var_df_infill_stns.csv')
#
#in_vgs_file = os.path.join(
#    r'santa_temperature_kriging_03',
#    r'temp_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')
#
#in_stns_coords_file = r'santa_temp_coords.csv'
#out_dir = r'santa_temperature_kriging_03'
#var_units = u'\u2103' #'centigrade'
#var_name = 'temperature'
#out_krig_net_cdf_file = r'santa_temp_kriging_%s_to_%s_5km.nc'
#==============================================================================

#==============================================================================
## PRECIPITATION
#in_data_file = os.path.join(
#        r'santa_infill_precipitation_Fabian_May2017_15_monthly_resamp',
#        r'infilled_var_df_infill_stns.csv')
#
#in_vgs_file = os.path.join(
#    r'santa_precipitation_kriging_01',
#    r'ppt_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')
#
#in_stns_coords_file = r'santa_ppt_coords_useable.csv'
#out_dir = r'santa_precipitation_kriging_02'
#var_units = 'mm'
#var_name = 'precipitation'
#out_krig_net_cdf_file = r'santa_ppt_kriging_%s_to_%s_5km.nc'
#==============================================================================

#==============================================================================
# POTENTIAL EVAPOTRANSPIRATION
in_data_file = r'santa_pet_may17.csv'

in_vgs_file = os.path.join(
    r'santa_pet_kriging_03',
    r'pet_fitted_variograms__nk_1.000__evg_name_robust__ngp_5__h_typ_var.csv')

in_stns_coords_file = r'santa_temp_coords.csv'
out_dir = r'santa_pet_kriging_03'
var_units = u'mm'
var_name = 'potential evapotranspiration'
out_krig_net_cdf_file = r'santa_pet_kriging_%s_to_%s_5km.nc'
#==============================================================================


strt_date = r'1963-01-01'
end_date = r'2015-12-31'

out_krig_net_cdf_file = out_krig_net_cdf_file % (strt_date, end_date)

# assuming in_drift_raster and in_stns_coords_file and in_bounds_shp_file
# have the same coordinates system
in_drift_raster = r'santa_rs_minerve_prep_may17/taudem_out/fil_5km.tif'
in_bounds_shp_file = r'santa_rs_minerve_prep_may17/taudem_out/watersheds.shp'

out_figs_dir = os.path.join(out_dir, 'krige_figs')

select_vg_lab = r'best_fit_vg_no_01'

freq = 'M'
x_coords_lab = 'X'
y_coords_lab = 'Y'
time_dim_lab = 'time'
nc_time_units = 'days since 1900-01-01 00:00:00.0'
nc_calendar = 'gregorian'

shp_color = 'k'

# cell width and height will be taken from the in_drift_raster if edk_krige
# is True
cell_width = 5000
cell_height = 5000

# interpolated values
min_var_val = 'min_in'
max_var_val = 'max_in'
#min_var_val = None
#max_var_val = None


in_sep = str(';')
in_date_fmt = '%Y-%m-%d'

plot_figs_flag = True

ord_krige_flag = True
sim_krige_flag = True
edk_krige_flag = True

#add_rand_noise = False
#ord_krige_flag = False
#sim_krige_flag = False
#edk_krige_flag = False

os.chdir(main_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if (not os.path.exists(out_figs_dir)) and plot_figs_flag:
    os.mkdir(out_figs_dir)

out_nc = nc.Dataset(os.path.join(out_dir, out_krig_net_cdf_file), mode='w')

#==============================================================================
# read the data frames
#==============================================================================
in_data_df = pd.read_csv(in_data_file,
                         sep=in_sep,
                         index_col=0,
                         encoding='utf-8')
in_vgs_df = pd.read_csv(in_vgs_file,
                        sep=in_sep,
                        index_col=0,
                        encoding='utf-8')
in_stns_coords_df = pd.read_csv(in_stns_coords_file,
                                sep=in_sep,
                                index_col=0,
                                encoding='utf-8')

in_data_df.dropna(inplace=True, axis=0, how='all')
in_vgs_df.dropna(inplace=True, axis=0, how='all')
in_stns_coords_df.dropna(inplace=True)

in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)
in_vgs_df.index = pd.to_datetime(in_vgs_df.index, format=in_date_fmt)

fin_date_range = pd.date_range(strt_date, end_date, freq=freq)
in_data_df = in_data_df.reindex(fin_date_range)
in_vgs_df = in_vgs_df.reindex(fin_date_range)

fin_stns = in_data_df.columns.intersection(in_stns_coords_df.index)
in_data_df = in_data_df.loc[:, fin_stns]
in_stns_coords_df = in_stns_coords_df.loc[fin_stns, :]

stns_min_x = in_stns_coords_df.loc[:, x_coords_lab].min()
stns_max_x = in_stns_coords_df.loc[:, x_coords_lab].max()
stns_min_y = in_stns_coords_df.loc[:, y_coords_lab].min()
stns_max_y = in_stns_coords_df.loc[:, y_coords_lab].max()

assert stns_min_x < stns_max_x, 'This cannot be!'
assert stns_min_y < stns_max_y, 'This cannot be!'

#==============================================================================
# Read the DEM
#==============================================================================
if edk_krige_flag:
    in_drift_ds = gdal.Open(in_drift_raster)
    assert in_drift_ds, 'GDAL cannot open %s' % in_drift_raster

    drift_rows = in_drift_ds.RasterYSize
    drift_cols = in_drift_ds.RasterXSize

    drift_geotransform = in_drift_ds.GetGeoTransform()
    _drift_x_min = drift_geotransform[0]
    _drift_y_max = drift_geotransform[3]

    drift_band = in_drift_ds.GetRasterBand(1)
    drift_ndv = drift_band.GetNoDataValue()

    cell_width = drift_geotransform[1]
    cell_height = abs(drift_geotransform[5])

    _drift_x_max = _drift_x_min + (drift_cols * cell_width)
    _drift_y_min = _drift_y_max - (drift_rows * cell_height)

    in_drift_arr = in_drift_ds.ReadAsArray()

#==============================================================================
# Read the bounding shapefile
#==============================================================================
in_ds = ogr.Open(in_bounds_shp_file)
lyr_count = in_ds.GetLayerCount()
assert lyr_count, 'No layers in %s!' % in_bounds_shp_file
assert lyr_count == 1, 'More than one layer in %s' % in_bounds_shp_file
in_lyr = in_ds.GetLayer(0)
envelope = in_lyr.GetExtent()
assert envelope, 'No envelope!'
in_ds.Destroy()

# get the boundary points for plotting
sf = shp.Reader(in_bounds_shp_file)
shp_xx = []
shp_yy = []

for shape in sf.shapeRecords():
    shp_xx.append([i[0] for i in shape.shape.points[:]])
    shp_yy.append([i[1] for i in shape.shape.points[:]])

fin_x_min, fin_x_max, fin_y_min, fin_y_max = envelope

fin_x_min -= 2*cell_width
fin_x_max += 2*cell_width
fin_y_min -= 2*cell_height
fin_y_max += 2*cell_height

if edk_krige_flag:
    assert fin_x_min > _drift_x_min
    assert fin_x_max < _drift_x_max
    assert fin_y_min > _drift_y_min
    assert fin_y_max < _drift_y_max

    min_col = int(max(0, (fin_x_min - _drift_x_min) / cell_width))
    max_col = int((fin_x_max - _drift_x_min) / cell_width)

    min_row = int(max(0, (_drift_y_max - fin_y_max) / cell_height))
    max_row = int((_drift_y_max - fin_y_min) / cell_height)
else:
    min_col = 0
    max_col = int((fin_x_max - fin_x_min) / cell_width)

    min_row = 0
    max_row = int((fin_y_max - fin_y_min) / cell_height)

#==============================================================================
# Calculate coordinates at which to krige

# with a buffer of one cell on all sides
#==============================================================================

assert 0 <= min_col <= max_col
assert 0 <= min_row <= max_row

strt_x_coord = fin_x_min + (0.5 * cell_width)
end_x_coord = fin_x_max

strt_y_coord = fin_y_max - (0.5 * cell_height)
end_y_coord = fin_y_min

krige_x_coords = np.arange(strt_x_coord,
                           end_x_coord,
                           cell_width)
krige_y_coords = np.arange(strt_y_coord,
                           end_y_coord,
                           -cell_height)
krige_x_coords_mesh, krige_y_coords_mesh = np.meshgrid(krige_x_coords,
                                                       krige_y_coords)
krige_coords_orig_shape = krige_x_coords_mesh.shape

if plot_figs_flag:
    # xy coords for pcolormesh
    pcolmesh_x_coords = np.linspace(fin_x_min,
                                    fin_x_max,
                                    (max_col - min_col + 1))

    pcolmesh_y_coords = np.linspace(fin_y_max,
                                    fin_y_min,
                                    (max_row - min_row + 1))


    krige_x_coords_plot_mesh, krige_y_coords_plot_mesh = \
        np.meshgrid(pcolmesh_x_coords, pcolmesh_y_coords)

krige_x_coords_mesh = krige_x_coords_mesh.ravel()
krige_y_coords_mesh = krige_y_coords_mesh.ravel()

if edk_krige_flag:

    krige_cols = np.arange(min_col, max_col, dtype=int)
    krige_rows = np.arange(min_row, max_row, dtype=int)

    assert krige_x_coords.shape[0] == krige_cols.shape[0]
    assert krige_y_coords.shape[0] == krige_rows.shape[0]
    krige_drift_cols_mesh, krige_drift_rows_mesh = np.meshgrid(krige_cols,
                                                               krige_rows)

    krige_drift_cols_mesh = krige_drift_cols_mesh.ravel()
    krige_drift_rows_mesh = krige_drift_rows_mesh.ravel()

    drift_vals = in_drift_arr[krige_drift_rows_mesh, krige_drift_cols_mesh]

    in_stns_drift_ser = pd.Series(index=in_stns_coords_df.index)

    for stn in in_stns_drift_ser.index:
        stn_x = in_stns_coords_df.loc[stn, x_coords_lab]
        stn_y = in_stns_coords_df.loc[stn, y_coords_lab]

        stn_col = int((stn_x - _drift_x_min) / cell_width)
        stn_row = int((_drift_y_max - stn_y) / cell_height)

        try:
            _ = in_drift_arr[stn_row, stn_col]
            if not np.isclose(drift_ndv, _):
                in_stns_drift_ser.loc[stn] = _
        except IndexError:
            pass

    in_stns_drift_ser.dropna(inplace=True)

#==============================================================================
# Krige
#==============================================================================

if ord_krige_flag:
    ord_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]), np.nan)

    for i, date in enumerate(fin_date_range):
        print '\n'
        curr_stns = in_data_df.loc[date, :].dropna().index

        assert curr_stns.shape == np.unique(curr_stns).shape

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, x_coords_lab].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, y_coords_lab].values

        model = unicode(in_vgs_df.loc[date, select_vg_lab])

        if curr_stns.shape[0] and (model != 'nan'):

            try:
                print 'OK,', date, ',', model, ',',
                print curr_stns.shape[0], 'stns'
                ord_krig = OrdinaryKriging(xi=curr_x_coords,
                                           yi=curr_y_coords,
                                           zi=curr_data_vals,
                                           xk=krige_x_coords_mesh,
                                           yk=krige_y_coords_mesh,
                                           model=model)
                ord_krig.krige()

                ord_krige_flds[i] = \
                    ord_krig.zk.reshape(krige_coords_orig_shape)

                print 'min:', ord_krige_flds[i].min(),
                print ', max:', ord_krige_flds[i].max()

                if min_var_val == 'min_in':
                    min_in = curr_data_vals.min()
                    ord_krige_flds[i][ord_krige_flds[i] < min_in] = min_in
                elif min_var_val is None:
                    pass
                elif (isinstance(min_var_val, float) or
                      isinstance(min_var_val, int)):
                    ord_krige_flds[i][ord_krige_flds[i] < min_var_val] = \
                        min_var_val
                else:
                    raise ValueError('Incorrect min_var_val specified!')

                if max_var_val == 'max_in':
                    max_in = curr_data_vals.max()
                    ord_krige_flds[i][ord_krige_flds[i] > max_in] = max_in
                elif max_var_val is None:
                    pass
                elif (isinstance(max_var_val, float) or
                      isinstance(max_var_val, int)):
                    ord_krige_flds[i][ord_krige_flds[i] > max_var_val] = \
                        max_var_val
                else:
                    raise ValueError('Incorrect max_var_val specified!')

                if plot_figs_flag:
                    plt.figure()
                    plt.pcolormesh(krige_x_coords_plot_mesh,
                                   krige_y_coords_plot_mesh,
                                   ord_krige_flds[i],
                                   vmin=np.nanmin(ord_krige_flds[i]),
                                   vmax=np.nanmax(ord_krige_flds[i]),
                                   alpha=1.0)
                    cb = plt.colorbar()
                    cb.set_label(var_name + ' (' + var_units + ')')
                    plt.scatter(curr_x_coords,
                                curr_y_coords,
                                label='obs. pts.',
                                marker='+',
                                c='r',
                                alpha=0.7)
                    plt.legend(framealpha=0.5)
                    for k in range(len(shp_xx)):
                        plt.plot(shp_xx[k], shp_yy[k], c=shp_color)

                    plt.xlabel('Easting (m)')
                    plt.ylabel('Northing (m)')

                    _ = 'Date: %s\n(vg: %s)\n' % (date.strftime('%Y-%m-%d'),
                                                  model)
                    plt.title(_)

                    plt.setp(plt.axes().get_xmajorticklabels(), rotation=70)
                    plt.axes().set_aspect('equal', 'datalim')
                    _ = os.path.join(out_figs_dir,
                                     'ok_%s.png' % date.strftime('%Y-%m-%d'))
                    plt.savefig(_, bbox_inches='tight')
                    plt.close()

            except Exception as msg:
                print 'Error:', msg

if sim_krige_flag:
    sim_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]), np.nan)

    for i, date in enumerate(fin_date_range):
        print '\n'
        curr_stns = in_data_df.loc[date, :].dropna().index

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, x_coords_lab].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, y_coords_lab].values

        model = unicode(in_vgs_df.loc[date, select_vg_lab])

        if curr_stns.shape[0] and (model != 'nan'):
            try:
                print 'SK,' , date,',' , model,',' ,
                print curr_stns.shape[0], 'stns'
                sim_krig = SimpleKriging(xi=curr_x_coords,
                                         yi=curr_y_coords,
                                         zi=curr_data_vals,
                                         xk=krige_x_coords_mesh,
                                         yk=krige_y_coords_mesh,
                                         model=model)
                sim_krig.krige()

                sim_krige_flds[i] = \
                    sim_krig.zk.reshape(krige_coords_orig_shape)

                print 'min:', sim_krige_flds[i].min(),
                print ', max:', sim_krige_flds[i].max()

                max_in = curr_data_vals.max()

                if min_var_val == 'min_in':
                    min_in = curr_data_vals.min()
                    sim_krige_flds[i][sim_krige_flds[i] < min_in] = min_in
                elif min_var_val is None:
                    pass
                elif (isinstance(min_var_val, float) or
                      isinstance(min_var_val, int)):
                    sim_krige_flds[i][sim_krige_flds[i] < min_var_val] = \
                        min_var_val
                else:
                    raise ValueError('Incorrect min_var_val specified!')

                if max_var_val == 'max_in':
                    sim_krige_flds[i][sim_krige_flds[i] > max_in] = max_in
                elif max_var_val is None:
                    pass
                elif (isinstance(max_var_val, float) or
                      isinstance(max_var_val, int)):
                    sim_krige_flds[i][sim_krige_flds[i] > max_var_val] = \
                        max_var_val
                else:
                    raise ValueError('Incorrect max_var_val specified!')

                if plot_figs_flag:
                    plt.figure()
                    plt.pcolormesh(krige_x_coords_plot_mesh,
                                   krige_y_coords_plot_mesh,
                                   sim_krige_flds[i],
                                   vmin=np.nanmin(sim_krige_flds[i]),
                                   vmax=np.nanmax(sim_krige_flds[i]),
                                   alpha=1.0)
                    cb = plt.colorbar()
                    cb.set_label(var_name + ' (' + var_units + ')')
                    plt.scatter(curr_x_coords,
                                curr_y_coords,
                                label='obs. pts.',
                                marker='+',
                                c='r',
                                alpha=0.7)
                    plt.legend(framealpha=0.5)
                    for k in range(len(shp_xx)):
                        plt.plot(shp_xx[k], shp_yy[k], c=shp_color)

                    plt.xlabel('Easting (m)')
                    plt.ylabel('Northing (m)')
                    _ = 'Date: %s\n(vg: %s)\n' % (date.strftime('%Y-%m-%d'),
                                                  model)
                    plt.title(_)

                    plt.setp(plt.axes().get_xmajorticklabels(), rotation=70)
                    plt.axes().set_aspect('equal', 'datalim')
                    _ = os.path.join(out_figs_dir,
                                     'sk_%s.png' % date.strftime('%Y-%m-%d'))
                    plt.savefig(_, bbox_inches='tight')
                    plt.close()
            except Exception as msg:
                print 'Error:', msg

if edk_krige_flag:
    edk_krige_flds = np.full((fin_date_range.shape[0],
                              krige_coords_orig_shape[0],
                              krige_coords_orig_shape[1]), np.nan)

    for i, date in enumerate(fin_date_range):
        print '\n'
        _ = in_data_df.loc[date, :].dropna().index
        curr_stns = _.intersection(in_stns_drift_ser.index)

        curr_data_vals = in_data_df.loc[date, curr_stns].values
        curr_x_coords = in_stns_coords_df.loc[curr_stns, x_coords_lab].values
        curr_y_coords = in_stns_coords_df.loc[curr_stns, y_coords_lab].values
        curr_drift_vals = in_stns_drift_ser.loc[curr_stns].values

        model = unicode(in_vgs_df.loc[date, select_vg_lab])

        if curr_stns.shape[0] and (model != 'nan'):
            try:
                print 'EDK,', date,',', model, ',',
                print curr_stns.shape[0], 'stns'
                edk_krig = ExternalDriftKriging(xi=curr_x_coords,
                                                yi=curr_y_coords,
                                                zi=curr_data_vals,
                                                si=curr_drift_vals,
                                                xk=krige_x_coords_mesh,
                                                yk=krige_y_coords_mesh,
                                                sk=drift_vals,
                                                model=model)
                edk_krig.krige()

                _ = edk_krig.zk.copy()
                _[np.isclose(drift_ndv, drift_vals)] = np.nan
                edk_krige_flds[i] = _.reshape(krige_coords_orig_shape)
                print 'min:', np.nanmin(edk_krige_flds[i]),
                print ', max:', np.nanmax(edk_krige_flds[i])

                if min_var_val == 'min_in':
                    min_in = curr_data_vals.min()
                    edk_krige_flds[i][edk_krige_flds[i] < min_in] = min_in
                elif min_var_val is None:
                    pass
                elif isinstance(min_var_val, float) or isinstance(min_var_val, int):
                    edk_krige_flds[i][edk_krige_flds[i] < min_var_val] = min_var_val
                else:
                    raise ValueError('Incorrect min_var_val specified!')

                if max_var_val == 'max_in':
                    max_in = curr_data_vals.max()
                    edk_krige_flds[i][edk_krige_flds[i] > max_in] = max_in
                elif max_var_val is None:
                    pass
                elif (isinstance(max_var_val, float) or
                      isinstance(max_var_val, int)):
                    edk_krige_flds[i][edk_krige_flds[i] > max_var_val] = \
                        max_var_val
                else:
                    raise ValueError('Incorrect max_var_val specified!')

                if plot_figs_flag:
                    plt.figure()
                    plt.pcolormesh(krige_x_coords_plot_mesh,
                                   krige_y_coords_plot_mesh,
                                   edk_krige_flds[i],
                                   alpha=1.0,
                                   vmin=np.nanmin(edk_krige_flds[i]),
                                   vmax=np.nanmax(edk_krige_flds[i]))
                    cb = plt.colorbar()
                    cb.set_label(var_name + ' (' + var_units + ')')
                    plt.scatter(curr_x_coords,
                                curr_y_coords,
                                label='obs. pts.',
                                marker='+',
                                c='r',
                                alpha=0.7)
                    plt.legend(framealpha=0.5)
                    for k in range(len(shp_xx)):
                        plt.plot(shp_xx[k], shp_yy[k], c=shp_color)

                    plt.xlabel('Easting (m)')
                    plt.ylabel('Northing (m)')
                    _ = 'Date: %s\n(vg: %s)\n' % (date.strftime('%Y-%m-%d'),
                                                  model)
                    plt.title(_)

                    plt.setp(plt.axes().get_xmajorticklabels(), rotation=70)
                    plt.axes().set_aspect('equal', 'datalim')
                    _ = os.path.join(out_figs_dir,
                                     'edk_%s.png' % date.strftime('%Y-%m-%d'))
                    plt.savefig(_, bbox_inches='tight')
                    plt.close()

            except Exception as msg:
                print 'Error:', msg

#==============================================================================
# Save outputs to netCDF files
#==============================================================================

out_nc.createDimension(x_coords_lab, krige_x_coords.shape[0])
out_nc.createDimension(y_coords_lab, krige_y_coords.shape[0])
out_nc.createDimension(time_dim_lab, fin_date_range.shape[0])

x_coords_nc = out_nc.createVariable(x_coords_lab,
                                    'd',
                                    dimensions=x_coords_lab)
x_coords_nc[:] = krige_x_coords

y_coords_nc = out_nc.createVariable(y_coords_lab,
                                    'd',
                                    dimensions=y_coords_lab)
y_coords_nc[:] = krige_y_coords


time_nc = out_nc.createVariable(time_dim_lab,
                                'd',
                                dimensions=time_dim_lab)
time_nc[:] = nc.date2num(fin_date_range.to_pydatetime(),
                         units=nc_time_units,
                         calendar=nc_calendar)
time_nc.units = nc_time_units
time_nc.calendar = nc_calendar
if ord_krige_flag:
    ok_nc = out_nc.createVariable('OK',
                                  'd',
                                  dimensions=(time_dim_lab,
                                              y_coords_lab,
                                              x_coords_lab))

    ok_nc[:] = ord_krige_flds
    ok_nc.units = var_units
    ok_nc.standard_name = var_name + ' (ordinary kriging)'

if sim_krige_flag:
    sim_nc = out_nc.createVariable('SK',
                                   'd',
                                   dimensions=(time_dim_lab,
                                               y_coords_lab,
                                               x_coords_lab))

    sim_nc[:] = sim_krige_flds
    sim_nc.units = var_units
    sim_nc.standard_name = var_name + ' (simple kriging)'

if edk_krige_flag:
    edk_nc = out_nc.createVariable('EDK',
                                   'd',
                                   dimensions=(time_dim_lab,
                                               y_coords_lab,
                                               x_coords_lab))

    edk_nc[:] = edk_krige_flds

    edk_nc.units = var_units
    edk_nc.standard_name = var_name + ' (external drift kriging)'

out_nc.Author = 'Faizan IWS Uni-Stuttgart'
out_nc.Source = out_nc.filepath()
out_nc.close()

STOP = timeit.default_timer()  # Ending time
print ('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START))
