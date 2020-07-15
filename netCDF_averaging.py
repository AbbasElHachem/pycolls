import netCDF4 as nc
import ogr
import numpy as np
import timeit
import faizpy10_linux
import os
import osr
#os.environ['GDAL_DATA'] = '/home/faizan/anaconda/pkgs/gdal-data-1.10.1-1/share/gdal'

print '\a\a\a\a Start \a\a\a\a\n'
start = timeit.default_timer() # to get the runtime of the program

in_nc_dir = r"I:\Stephen\to_convert"
out_nc_dir = r"I:\Stephen\converted_daily\second_run/"

in_nc_list = faizpy10_linux.list_full_path('.nc', in_nc_dir)
print 'These file(s) will be processed: %s\n' % in_nc_list

# location of the shapefile that will be used to get the extents and
# finally the references of the cells that fall with the the extents
in_shp_dir = r"I:\Stephen\ext_shps"

# the interval (hours in this case) at which
# we need to accumulate/sum/average the data in the netCDF file
intvl = 8
nc_EPSG = 4326

# names of the different variables in the netCDf file(s)
in_lat_name = 'lat'
in_lon_name = 'lon'
in_ppt_name = 'pr'
in_temp_name = 'tas'
in_time_name = 'time'

# names of the variables for the output netCDf file(s)
out_ppt_name = 'pr_daily'
out_temp_name = 'tas_mean'
out_time_name =  in_time_name

# units for the various variables of the output netCDF files(s)
out_ppt_units = 'mm_per_day'
out_temp_units = 'C' # 273.15 is subtracted from all values
#out_time_units = 'ordinal_hr'
#out_lat_units = out_lon_units = 'degree'

# read the input shapefile to get the extents
in_shps = faizpy10_linux.list_full_path('.shp', in_shp_dir)

xmin, xmax, ymin, ymax = 999., -999., 999., -999.
for shp in in_shps:
    in_vec = ogr.Open(shp, 0)
    in_lyr = in_vec.GetLayer()
    ext = in_lyr.GetExtent()
    spt_ref = in_lyr.GetSpatialRef()
    trgt = osr.SpatialReference()
    trgt.ImportFromEPSG(nc_EPSG)
    tfm = osr.CreateCoordinateTransformation(spt_ref, trgt)

    xmin_ll, ymin_ll, z = tfm.TransformPoint(ext[0], ext[2])
    xmax_ul, ymax_ul, z = tfm.TransformPoint(ext[1], ext[3])

    in_vec.Destroy()

    if xmin_ll < xmin:
        xmin = xmin_ll
    if xmax_ul > xmax:
        xmax = xmax_ul
    if ymin_ll < ymin:
        ymin = ymin_ll
    if ymax_ul > ymax:
        ymax = ymax_ul


for in_nc_file in in_nc_list:
    # location of the input netCDf file(s)
    #in_nc_file = r"I:\Stephen\wa12clmN_echam6_hist_fdda_mpp_1979_1990\wrfsfc\wa12clmN_echam6-hist_wrfsfc_d01_1979-01_1989-12.nc"

    # location of the output netCDF files(s)
    out_nc_file = out_nc_dir + os.path.basename(in_nc_file)[:-3] + '_daily.nc'
    print '\n\nOutput netCDF is: %s\n' % out_nc_file

    # read the input netCDF and create the output netCDF
    in_nc = nc.Dataset(in_nc_file)
    out_nc = nc.Dataset(out_nc_file, mode='w')

    # get the various variables as arrays.
    lons = in_nc.variables[in_lon_name][:] # shape is (455,)

    lats = in_nc.variables[in_lat_name][:] # shape is (282,)

    time = in_nc.variables[in_time_name] # shape is (32144,)

    # get the indicies of the minimum and maximum extents in the netCDF file
    min_lon_idx = np.argmin(np.abs(lons-xmin), axis=0)
    max_lon_idx = np.argmin(np.abs(lons-xmax), axis=0)

    min_lat_idx = np.argmin(np.abs(lats-ymin), axis=0)
    max_lat_idx = np.argmin(np.abs(lats-ymax), axis=0)

    # define the final arrays for the output
    fin_lat_arr = lats[min_lat_idx:max_lat_idx]
    fin_lon_arr = lons[min_lon_idx:max_lon_idx]
    fin_ppt_arr = np.empty((0, fin_lat_arr.shape[0], fin_lon_arr.shape[0]))
    fin_temp_arr = np.empty((0, fin_lat_arr.shape[0], fin_lon_arr.shape[0]))
    fin_time_arr = np.empty((0))

    # total number of layers (the 3 hr data) in the input netCDF
    tot_dims = in_nc.variables[in_ppt_name].shape[0]

    loop_count = 0 # loop counter

    big_step = 8048 # how many step can be read from the netCDF at once
    # this is done because the netCDF array can be too big for memory
    # should be a multiple of the variable 'intvl'

    # adjusting the range of the for-loop so that it runs correctly
    loop_range = tot_dims - (tot_dims%big_step) + big_step

    for j in range(0, loop_range, big_step):

        loop_count += 1
        print '\a\a\a\a Going through loop number: %d' % loop_count

        # adjust the last index if it gets bigger than 'tot_dims'
        if j+big_step > tot_dims:
            big_step = tot_dims - j

        # read a chunk of variables in the netCDF file(s)
        ppt_big_slice = in_nc.variables[in_ppt_name][j:(j+big_step+1),
                            min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx]

        temp_big_slice = in_nc.variables[in_temp_name][j:(j+big_step+1),
                            min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx]

        time_big_slice = time[j:(j+big_step+1)]

        print 'Outside-loop indices were: %s, %s' % (j, j+big_step+1)

        # this for-loop converts the 3-hr data into daily
        for i in range(0, big_step, intvl):
            try:
                pass
                acc_ppt_init = ppt_big_slice[i]
                acc_ppt_fin = ppt_big_slice[(i+intvl)]
                temp_small_slice = temp_big_slice[i:i+intvl+1]
                # taking: the last time step (oridinal) only
                lst_time_slice = time_big_slice[(i+intvl)]
            except IndexError:
                pass
                arr_size = time_big_slice.shape[0]
                print "\nIndex Error occured. Indices were: %d, %d" % (i, i+intvl)
                print "Total size of array is: %d" % arr_size
                print 'Trying to adjust the last index by 1...'
                if i+intvl == arr_size:
                    k  = arr_size - 1
                    acc_ppt_init = ppt_big_slice[i]
                    acc_ppt_fin = ppt_big_slice[k]
                    temp_small_slice = temp_big_slice[i:k]
                    # taking: the last time step (oridinal) only
                    lst_time_slice = time_big_slice[k]
                    print 'Everything looks good...'
                else:
                    print "Check the script for index values."
                    print "Adjusting the last index by 1 did not work."
                    quit()
            #subtracting: assuming 'pr' is cummulative over the whole time
            acc_ppt_slice = np.subtract(acc_ppt_fin, acc_ppt_init)
            fin_ppt_arr = np.concatenate((fin_ppt_arr, acc_ppt_slice[np.newaxis]))

            # taking mean: assuming 'tas' is the for every three hours
            acc_temp_slice = np.mean(np.subtract(temp_small_slice, 273.15), axis=0)
            fin_temp_arr = np.concatenate((fin_temp_arr, acc_temp_slice[np.newaxis]))
            fin_time_arr = np.append(fin_time_arr, lst_time_slice)

        print 'Last inside-loop indices were: %s, %s\n' % (i, i+intvl)


    print 'ppt array shape: ', fin_ppt_arr.shape
    print 'temp array shape: ', fin_temp_arr.shape
    print 'time array shape: ', fin_time_arr.shape
    print 'lon array shape: ', fin_lon_arr.shape
    print 'lat array shape: ', fin_lat_arr.shape

    # create the dimensions of various variables
    # in the output netCDF
    out_nc.createDimension(out_ppt_name,
                           None)

    out_nc.createDimension(out_temp_name,
                           None)

    out_nc.createDimension(out_time_name,
                           None)

    out_nc.createDimension(in_lat_name,
                           fin_lat_arr.shape[0])

    out_nc.createDimension(in_lon_name,
                           fin_lon_arr.shape[0])

    # create the different variables in the output
    # netCDF
    out_ppt = out_nc.createVariable(out_ppt_name,
                                 'd',
                                 dimensions=(
                                 out_time_name,
                                 in_lat_name,
                                 in_lon_name)
                                 )

    out_temp = out_nc.createVariable(out_temp_name,
                                 'd',
                                 dimensions=(
                                 out_time_name,
                                 in_lat_name,
                                 in_lon_name)
                                 )

    out_time = out_nc.createVariable(out_time_name,
                                 'd',
                                 (out_time_name)
                                 )

    out_lats = out_nc.createVariable(in_lat_name,
                                 'd',
                                 (in_lat_name)
                                 )

    out_lons = out_nc.createVariable(in_lon_name,
                                 'd',
                                 (in_lon_name)
                                 )

    # assign values to the different
    # variables in the output netCDF
    out_ppt[:] = fin_ppt_arr
    out_temp[:] = fin_temp_arr
    out_time[:] = fin_time_arr
    out_lats[:] = fin_lat_arr
    out_lons[:] = fin_lon_arr

    # name the units
    out_ppt.units = out_ppt_units
    out_ppt.standard_name = 'precipitation_amount'
    out_ppt.long_name = 'daily_precipitation'
    #out_ppt._ChunkSizes = 1,1,1

    out_temp.units = out_temp_units
    out_temp.standard_name = 'air_temperature'
    out_temp.long_name = 'Near-Surface air temperature'
    #out_temp._ChunkSizes = 1,1,1

    out_time.units = "hours since 1970-01-01 00:00:00 +0000"
    out_time.standard_name = 'time'
    out_time.long_name = 'Time'
    out_time.calender = 'standard'
    #out_time._ChunkSizes = 1


    out_lons.units = 'degrees_east'
    out_lons.standard_name = 'longitude'
    out_lons.long_name = 'Longitude, west is negative'
    out_lons.axis = 'X'

    out_lats.units = 'degrees_north'
    out_lats.standard_name = 'latitude'
    out_lats.long_name = 'Latitude, south is negative'
    out_lats.axis = 'Y'
    # assign some other stuff (not necessary)
    out_nc.Author = in_nc.Author + ' and Faizan_HIWI'
    out_nc.History = '8 hour data converted to daily'
    out_nc.Source = in_nc.filepath()
    out_nc.Title = in_nc.Title


    try:
        in_nc.close()
        out_nc.close()
    except:
        print 'Could not close the netCDF files.'

endtime = timeit.default_timer() - start
print '\n\a\a\a Done with everything. Total run time was about %s seconds \a\a\a' % (round(endtime, 1))
