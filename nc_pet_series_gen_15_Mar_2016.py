# Define the required modules here
import netCDF4 as nc
import ogr
import osr
import numpy as np
import timeit
import datetime as dt
import pandas as pd
import sys
from faizpy import list_all_full_path, create_new_dir
import os
import fnmatch

print 'To extract data at the points given in a shapefile from a netCDF file.'
print 'Running script....'

#==============================================================================
# # Define the required directories and shapefiles here
#==============================================================================

# directory containing input netCDF files
in_nc_dir = r"D:\POTEVP_RCMs/" # everything created inside this directory
# directory for output station data storage
out_csv_dir = r"Potential_Evapotranspiration_data/"
# coordinate sys doesnt matter (a point shapefile)
in_ppt_shp = r"Shps\Precip_st.shp"
# coordinate sys doesnt matter (a point shapefile)
in_temp_shp = r"Shps\Temp_st.shp"
# a text file to store all the activity output in
hist_file_loc = r"history_stephen_daily_pet_data_15_mar_2016.txt"

#==============================================================================
# # Define some more stuff
#==============================================================================
# output csv prefixes
out_ppt_pref = 'pet__ppt_stns_'
out_temp_pref = 'pet__temp_stns_'

# input and output time format
# I am assuming that the units variable of time reads: 'days since date_time_fmt'
date_time_fmt = '%Y-%m-%d %H:%M:%S' # after the 'days since' thing...

# NOTE:
# ppt = precipitation
# temp = temperature

# list variables names in the netCDF files
in_ppt_name_1 = 'evspsblpot' # the ppt name in the netCDF files
in_temp_name_1 = 'evspsblpot' # the temp name in the netCDF files
in_time_name = 'time' # time variable
in_lat_name = 'lat' # latitude variable
in_lon_name = 'lon' # longitude variable
nc_EPSG = 4326 # EPSG code of the input netCDF coordinate system

#==============================================================================
# # Start computing...
#==============================================================================
os.chdir(in_nc_dir)

# just something to write the screen output to a file. For later viewing.
saveout = sys.stdout
fsock = open(hist_file_loc, 'w')
sys.stdout = fsock

print '\a\a\a\a Start \a\a\a\a\n'
start = timeit.default_timer() # to get the runtime of the program
print '\a\a\a Main starting time is: ', dt.datetime.now(), '\n\n\n'

# read the ppt stations shapefile and create a
# coordinate transformation between the shapefile and the netCDF
in_ppt_ds = ogr.Open(in_ppt_shp)
ppt_lyr = in_ppt_ds.GetLayer()
ppt_spt_ref = ppt_lyr.GetSpatialRef()
ppt_trgt = osr.SpatialReference()
ppt_trgt.ImportFromEPSG(nc_EPSG)
ppt_tfm = osr.CreateCoordinateTransformation(ppt_spt_ref, ppt_trgt)

# read the temp stations shapefile and create a
# coordinate transformation between the shapefile and the netCDF
in_temp_ds = ogr.Open(in_temp_shp)
temp_lyr = in_temp_ds.GetLayer()
temp_spt_ref = temp_lyr.GetSpatialRef()
temp_trgt = osr.SpatialReference()
temp_trgt.ImportFromEPSG(nc_EPSG)
temp_tfm = osr.CreateCoordinateTransformation(temp_spt_ref, temp_trgt)

create_new_dir(out_csv_dir) # create the output directory


# go through each file in the given input directory and
# preform the required actions

all_ncs = list_all_full_path('.nc', in_nc_dir)
print 'Total number of files to process: %d' % len(all_ncs)

for nc_no, in_nc_file in enumerate(all_ncs):
    print 'Going through: \n%d). %s, at time: %s' % (nc_no + 1, in_nc_file, dt.datetime.now())

    # read the netCDF and get the required inputs
    in_nc = nc.Dataset(in_nc_file)
    lat_arr = in_nc.variables[in_lat_name][:]
    lon_arr = in_nc.variables[in_lon_name][:]

    # convert the netCDF time to regular time
    time = in_nc.variables[in_time_name]
    time_arr =  nc.num2date(time[:], time.units, calendar=time.calendar)

    print 'Counting time from (in the netCDF file):',time.units
    print 'Start date in the netCDF: ', time_arr[0]
    print 'End date in the netCDF: ', time_arr[-1]
    print 'Total time steps in the netCDF: ', time_arr.shape[0], '\n\n'
#==============================================================================
# writing precipitation data
#==============================================================================
    try:
        vals = in_nc.variables[in_ppt_name_1] # check if the variable exists in the netCDF file
        try:
            # create a pandas DataFrame to save all the data in
            main_ppt_df = pd.DataFrame(index=time_arr)

            # go through each point in the shapefile, extract the data
            ppt_lyr.ResetReading()
            ppt_feat = ppt_lyr.GetNextFeature()
            ppt_stn_list = []
            ppt_lat_idx_list = []
            ppt_lon_idx_list = []
            ppt_x_list = []
            ppt_y_list = []
            while ppt_feat:
                stn_name = ppt_feat.GetField(4)
                ppt_stn_list.append(stn_name)
                print 'Precipitation station name is: %s' % stn_name

                ppt_x = ppt_feat.geometry().GetX()
                ppt_y = ppt_feat.geometry().GetY()

                print 'Original coordinates: %s, %s' % (ppt_x, ppt_y)

                ppt_tfm_pts = ppt_tfm.TransformPoint(ppt_x, ppt_y)

                print 'Transformed coordinates: %s, %s' % (ppt_tfm_pts[0], ppt_tfm_pts[1])

                lat_idx = np.argmin(np.abs(lat_arr - ppt_tfm_pts[1]), axis=0)[0]
                lon_idx = np.argmin(np.abs(lon_arr - ppt_tfm_pts[0]), axis=1)[0]

                print 'Indices in the netCDF file are: %s, %s\n' % (lon_idx, lat_idx)

                ppt_x_list.append(lon_arr[0, lon_idx])
                ppt_y_list.append(lat_arr[lat_idx, 0])

                ppt_lat_idx_list.append(lat_idx)
                ppt_lon_idx_list.append(lon_idx)

                ppt_feat = ppt_lyr.GetNextFeature()

                try:
                    series = vals[:, lat_idx, lon_idx]
                    main_ppt_df[stn_name + '_per_sec'] = series
                    main_ppt_df[stn_name + '_per_day'] = series * 86400
                except IndexError:
                    print IndexError('Something wrong while reading indcies.')

            out_ppt_csv = out_csv_dir + out_ppt_pref + os.path.basename(in_nc_file).split('.')[0] + '.csv'
            print '\nOutput .csv file is:\n%s\n\n' % (out_ppt_csv)

            # drop the records in the DataFrame that have no values
            main_ppt_df.dropna(inplace=True)
            # save the DataFrame to a csv file
            main_ppt_df.to_csv(out_ppt_csv, sep=';', float_format='%.9f',
                               date_format=date_time_fmt)

        except RuntimeError:
            print RuntimeError('Couldn\'t write the precipitation data to output file.')

    except KeyError:
        print KeyError('No precipitation variable')

#==============================================================================
# writing temperature data
#==============================================================================
    try:
        vals = in_nc.variables[in_temp_name_1][:]# Check if the variable exists in the netCDF file
        try:
            # create a pandas DataFrame and save all the data in it
            main_temp_df = pd.DataFrame(index=time_arr)

            # go through each point in the shapefile, extract the data
            temp_lyr.ResetReading()
            temp_feat = temp_lyr.GetNextFeature()
            temp_stn_list = []
            temp_lat_idx_list = []
            temp_lon_idx_list = []
            temp_x_list = []
            temp_y_list = []
            while temp_feat:
                stn_name = temp_feat.GetField(4)
                temp_stn_list.append(stn_name)
                print 'Temperature station name is: %s' % stn_name

                temp_x = temp_feat.geometry().GetX()
                temp_y = temp_feat.geometry().GetY()

                print 'Original coordinates: %s, %s' % (temp_x, temp_y)

                temp_tfm_pts = temp_tfm.TransformPoint(temp_x, temp_y)

                print 'Transformed coordinates: %s, %s' % (temp_tfm_pts[0], temp_tfm_pts[1])

                lat_idx = np.argmin(np.abs(lat_arr - temp_tfm_pts[1]), axis=0)[0]
                lon_idx = np.argmin(np.abs(lon_arr - temp_tfm_pts[0]), axis=1)[0]

                print 'Indices in the netCDF file are: %s, %s\n' % (lon_idx, lat_idx)

                temp_x_list.append(lon_arr[0, lon_idx])
                temp_y_list.append(lat_arr[lat_idx, 0])

                temp_lat_idx_list.append(lat_idx)
                temp_lon_idx_list.append(lon_idx)

                temp_feat = temp_lyr.GetNextFeature()

                try:
                    series = vals[:, lat_idx, lon_idx]
                    main_temp_df[stn_name + '_per_sec'] = series
                    main_temp_df[stn_name + '_per_day'] = series * 86400
                except IndexError:
                    print IndexError('Something wrong while reading indices.')

            out_temp_csv = out_csv_dir + out_temp_pref + os.path.basename(in_nc_file).split('.')[0] + '.csv'
            print '\nOutput .csv file is:\n%s\n\n' % (out_temp_csv)

            # drop the records in the DataFrame that have no values
            main_temp_df.dropna(inplace=True)
            # save the DataFrame to a csv file
            main_temp_df.to_csv(out_temp_csv, sep=';', float_format='%.5f',
                                date_format=date_time_fmt)

        except RuntimeError:
            print RuntimeError('Couldn\'t write the temperature data to output file.')

    except KeyError:
        print KeyError('No temperature variable')

    # close the netCDF file
    in_nc.close()
    print 'Done with: \n%s, at: %s' % (in_nc_file, dt.datetime.now())
    print '\n\n\n\n'
# close the input shapefiles finally
in_ppt_ds.Destroy()
in_temp_ds.Destroy()

print 'Precipitation stations with their indices and coordinates in the netCDF e.g. (name, row, col, x, y)'
print zip(ppt_stn_list, ppt_lat_idx_list, ppt_lon_idx_list, ppt_x_list, ppt_y_list)

print '\n\n\n'

print 'Temperature stations with their indices and coordinates in the netCDF e.g. (name, row, col, x, y)'
print zip(temp_stn_list, temp_lat_idx_list, temp_lon_idx_list, temp_x_list, temp_y_list)

# compute total runtime in seconds
print '\a\a\a Main ending time is: ', dt.datetime.now(), '\n\n\n'
endtime = timeit.default_timer() - start
print '\n\a\a\a Total run time was about %s seconds \a\a\a' % (round(endtime, 3))

# save the terminal output to the history file
sys.stdout = saveout
fsock.close()

#==============================================================================
print 'Done with everything.'
#==============================================================================
