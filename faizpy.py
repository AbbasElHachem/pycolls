"""My functions
"""

from math import sin, cos, tan, acos, radians, pi, sqrt
import os
import sys
import subprocess
import math
import linecache
from fnmatch import fnmatch
#import rasterstats
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import gdal, ogr, osr
import scipy.ndimage as spni
from numpy import linspace
from numpy import meshgrid
import matplotlib as mpl
import numpy as np
import pyproj
import pandas as pd
#from numba.decorators import jit
import timeit

#ogr.UseExceptions()
#osr.UseExceptions()
#gdal.UseExceptions()

os_geo_dir = os.path.dirname(sys.executable)

warp_file = os.path.join(os_geo_dir, "Lib\site-packages\osgeo\gdalwarp.exe") #full path to the \
                                                    # GDALWARP executable
translate_file= os.path.join(os_geo_dir, "Lib\site-packages\osgeo\gdal_translate.exe") #full path to \
                                                    # the GDALTRANSLATE executable
rasterize_file = os.path.join(os_geo_dir,  "Lib\site-packages\osgeo\gdal_rasterize.exe") # gdal executable \
                                                    # that converts vector data to raster data
polygonize_file = os.path.join(os_geo_dir, "Scripts\gdal_polygonize.py") # gdal script \
                                                    # that converts raster data to vector data

def list_files(ext, file_dir):
    """
    Purpose: To return file names in a given folder with a given extension in \
                ascending order.
    Description of the arguments:
        ext (string) = Extension of the files to choose for listing \
                e.g. '.txt', '.tif'.
        file_dir (string) = Full path of the to the folder in which the files \
                reside in string format.
    """
    new_list = []
    for elm in os.listdir(file_dir):
        if elm[-len(ext):] == ext:
            new_list.append(elm)
    return(sorted(new_list))


def list_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in a given folder with a \
            given extension in ascending order.
    Description of the arguments:
        ext (string) = Extension of the files to list \
            e.g. '.txt', '.tif'. \n\n
        file_dir (string) = Full path of the folder in which the files \
            reside.
    """
    new_list = []
    for elm in os.listdir(file_dir):
        if elm[-len(ext):] == ext:
            new_list.append(file_dir + '/' + elm)
    return(sorted(new_list))


def list_all_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in all dirs of a given folder with a \
            given extension in ascending order.
    Description of the arguments:
        ext (string) = Extension of the files to list \
            e.g. '.txt', '.tif'.
        file_dir (string) = Full path of the folder in which the files \
            reside.
    """
    new_list = []
    patt = '*' + ext
    for root, dirc, files in os.walk(file_dir):
        for elm in files:
            if fnmatch(elm, patt):
                full_path = os.path.join(root, elm)
                new_list.append(full_path)
    return(sorted(new_list))


def create_new_dir(new_path, quiet=True):
    """
    Purpose: To create a new directory
    Description of the arguments:
        new_path (string): Full path of the directory to create \
                    e.g. 'C:/new_path' will create a folder new_path in C:
        Note: if the path exists already then the function does nothing and \
                    returns a message showing that the directory exists already
    """
    if not os.path.exists(new_path):
        if quiet==False:
            print('Creating Directory: ', new_path)
        os.mkdir(new_path)
    else:
        if quiet==False:
            print('This Directory exists: ', new_path)
    return


def get_ras_props(in_ras, in_band_no=1):
    """
    Purpose: To return a given raster's extents, number of rows and columns, \
                        pixel size in x and y direction, projection, noData value, \
                        band count and GDAL data type using GDAL as a list.
    Description of the arguments:
        in_ras (string): Full path to the input raster. If the raster cannot be \
                        read by GDAL then the function returns None.
        in_band_no (int): The band which we want to use (starting from 1). \
                        Defaults to 1. Used for getting NDV.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds is not None:
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize

        geotransform = in_ds.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        x_max = x_min + (cols*pix_width)
        y_min = y_max - (rows*pix_height)

        proj = in_ds.GetProjectionRef()

        in_band = in_ds.GetRasterBand(in_band_no)
        if in_band is not None:
            NDV = in_band.GetNoDataValue()
            gdt_type = in_band.DataType
        else:
            NDV = None
            gdt_type = None
        band_count = in_ds.RasterCount

        ras_props = [x_min, x_max, y_min, y_max, cols, rows, pix_width, \
                     pix_height, proj, NDV, band_count, gdt_type]

        in_ds = None
        return ras_props
    else:
        print('\n\a Could not read the input raster (%s). Check path and file!' % in_ras)
        return


# Need to change this. Layer and Features in a layer are different
def get_vec_props(inVecFile, layerIndex=None, layerName=None):
    """
    Purpose: To return the extents and spatial reference of a given layer \
                    (point, line, polygon) in a given ogr vector file as a list.
    Description of the arguments:
        inVecFile (string): Full path to the vector file.
        layerIndex (int): The index of the layer(starting from 0) \
                    inside the vector file. The user can specify either \
                    layerIndex or layerName.
        layerName (string): The name of the layer inside the vector file to get the properties for.
        Note: if the specified types of arguments are not used then None returned. \
                replaces the lyrNameAsStr(vecFilePath) function.
    """
    inVec = ogr.Open(inVecFile, 0)
    if inVec is not None:
        if type(layerIndex) == type(int()):
            layer = inVec.GetLayer(layerIndex)
        elif type(layerName) ==  type(str()):
            layer = inVec.GetLayerByName(layerName)
        else:
            print('\a Cannot get the specified layer in %s.\
                        \n\a The layerIndex argument should be of integer type or the \
                        layerName argument should have string type.')
            return None
        if layer is not None:
            geotransform = layer.GetExtent()
            xMin = geotransform[0]
            yMax = geotransform[3]
            xMax = geotransform[1]
            yMin = geotransform[2]
            spRef = layer.GetSpatialRef()
            vecProps = [xMin, xMax, yMin, yMax, spRef]
            inVec = None
            return vecProps
        else:
            print('\a Could not read the input vector file (%s) for the given \
                        layer index or layer name.' % inVecFile)
            return None


def reproject_raster_GDALWARP(in_ras,
                              out_ras,
                              out_epsg,
                              in_band_no=1,
                              no_data_value='input',
                              warp_file=warp_file,
                              res_meth='near',
                              comp_type='LZW',
                              out_driver='GTiff'):
    """
    Purpose: To reproject a given raster into a new one given a new coordinate system.

    Description of the argument:

        in_ras (string): Full path to the input raster.

        out_ras (string): Full path of the ouptut raster.

        out_epsg (int): EPSG of output coordinate system.

        in_band_no (int): The band which we want to reproject (starting from 1). \
                Only one band can be reprojected. Defaults to 1.

        no_data_value (int or float): The output no_data value. If the user knows \
                then it's better to specify it here.

        warp_file (string): Full path to the GDALWARP executable. Specified in the \
                begining of this script.

        res_meth (string): The resampling method to use while reprojecting. \
                    Can be 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', \
                    'average', 'mode'. Defaults to 'near' which requires the least \
                    computational effort with worst accuracy for continuous data. \
                    It is good for categorical data though.

        comp_type (string): The type of compression to use while creating the output raster.\
                    Can be 'LZW', 'DEFLATE', 'PACKBITS'.

        out_driver (string): Short name of the output driver according to GDAL
    """
    out_spref = osr.SpatialReference()
    out_spref.ImportFromEPSG(out_epsg)
    out_spref_str = out_spref.ExportToWkt()

    in_ds = gdal.Open(in_ras, 0)
    if in_ds is None:
        print('Could not open input raster (%s)' % in_ras)
        return
    else:
        proj_ds = in_ds.GetProjection()
        proj_str = osr.GetUserInputAsWKT(proj_ds)
        if no_data_value == 'input':
            try:
                NDV = in_ds.GetRasterBand(int(in_band_no)).GetNoDataValue()
            except:
                print('\n\a Could not get the noData value from the input \
                        raster (%s). Specify manually!' % in_ras)
                return
        else:
            NDV = no_data_value
        in_ds = None

        arg = warp_file.replace("/", "\\"), ' -multi -s_srs ' + proj_str + ' -r ' + res_meth + \
                        ' -t_srs ' + out_spref_str  + ' -srcnodata ' + str(NDV) + \
                        " -co COMPRESS=" + comp_type + ' -dstnodata ' + str(NDV) + " -of " + out_driver + \
                        ' -overwrite -q ' + in_ras.replace("/", "\\") + " " + out_ras.replace("/", "\\")
        subprocess.call([arg])

        return


def shpReProj_Polygon(inDs, outDs, out_EPSG, layerName):
    """
    Purpose: To reproject a given ESRI shapefile using OGR
    Description of the arguments:
        inDs (string): Full path to the input dataset.
        outDs (string): Full path of the output dataset.
        out_EPSG: EPSG code of the output projection. \
                    Can be had from spatialreference.org
        layerName (string): The layer name of the output polygons.
        Note: This function was only tested for a shapefile with only one polygon.
                    So care should be taken for multiple polygon shapefile.
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # input SpatialReference
    vec = ogr.Open(inDs,0)
    inLayer = vec.GetLayer()
    inSpatialRef = inLayer.GetSpatialRef()

    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(out_EPSG)
    outSpatialRef.MorphToESRI()

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # create the output layer
    outputShapefile = outDs
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer(layerName, geom_type=ogr.wkbMultiPolygon)

    outPrj = open((outDs.rsplit('.',1)[0] + '.prj'), 'w')
    outPrj.write(outSpatialRef.ExportToWkt())
    outPrj.close()

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    vec.Destroy()
    outDataSet.Destroy()
    return


def mosaic_ras_GDAL(in_ras_list, out_ras, no_data_value='input',
                    mosaic_type='last', temp_ras=r"D:\temp_ras",
                    any_NDV_to_NDV=False, out_driver='GTiff',
                    out_data_type='input', prec=6, in_band_no=1,
                    comp_type='None'):
    """
    Purpose: To mosaic a given list of rasters into one. \
    Description of the arguments:
        in_ras_list(list containing strings): The list of input rasters to be mosaiked. \
                    Each item in the list is composed of the full path of the raster \
                    e.g. 'c:\temp_ras.tif'.
        out_ras (string): Full path of the output mosaik raster. \
                    The driver to create the out_ras is same as the first input raster.
        no_data_value (int or float): The noData value of the out_ras. Can be read from \
                    the first input raster if the argument is 'input' otherwise \
                    the user can specify a value.
        mosaic_type (string): Currently, there are three types of mosaic techniques \
                    available. They are:
                    'last': The value that the mosaic raster will have, when two \
                            or more rasters overlap, is that of the last raster \
                            in the in_ras_list.
                    'first': Same as the last but the mosaic value is that from \
                            the first raster in the in_ras_list.
                    'average': As the name suggests it is the average of the \
                            overlapping cell values. The output value depends \
                            on the datatype of the first input raster in in_ras_list.
                    'minimum': Minimum cell value of all the overlapping cells at a point.
                    'maximum': Maximum cell value of all the overlapping cells at a point. \

        temp_ras (string): It is a raster that contains the count of the number of rasters \
                    that overlap at a given pixel e.g. if a pixel has three rasters \
                    overlapping on it then its value is three. \
                    This raster is only created if the mosaic type is 'average' (till now). \
                    It is deleted in the end. It has the same driver as the first input raster.
        any_NDV_to_NDV (boolean): if True, Set value at a pixel to NDV if any raster \
                    has NDV at that pixel.
        out_driver (string, short format name): The output driver of the out_ras. \
                    Defaults to 'GTiff'.
        out_data_type (int or float): The dataType of the output. Defaults to the \
                    dataType of the first raster in the in_ras_list.
        prec (int): The precision to use while matching the extents and cell size of rasters.
        in_band_no (int): The band number to use in the input raster. only one band can be used.
        comp_type (string):The type of compression to use while creating the output raster.\
                    Can be 'LZW', 'DEFLATE', 'PACKBITS'.
        Note: A check is performed on all the raster for same cell size and \
                    coordinate system. If any raster fails to match the others the function returns.

    """
    # Defining empty lists to store rasters' extents and other stuff
    # These lists are used later on
    x_min_list = []
    x_max_list = []
    y_min_list = []
    y_max_list = []
    pix_width_list = []
    pix_height_list = []
    rows_list = []
    cols_list = []

    # Appending values to lists
    for in_ras in in_ras_list:
        in_ds = gdal.Open(in_ras, 0)
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize
        geotransform = in_ds.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]
        pix_width = geotransform[1]
        pix_height = geotransform[5]
        x_max = x_min + (cols*pix_width)
        y_min = y_max + (rows*pix_height)
        x_min_list.append(round(x_min, prec))
        x_max_list.append(round(x_max, prec))
        y_min_list.append(round(y_min, prec))
        y_max_list.append(round(y_max, prec))
        pix_width_list.append(pix_width)
        pix_height_list.append(pix_height)
        rows_list.append(rows)
        cols_list.append(cols)
    in_ds = None

    # Choosing extents of the output mosaic raster
    out_ras_x_min = min(x_min_list)
    out_ras_x_max = max(x_max_list)
    out_ras_y_min = min(y_min_list)
    out_ras_y_max = max(y_max_list)

    # Checking to see pixel if widths/heights are consistent for all the input \
        # rasters, if not the function exits
    for i in range(0, len(pix_width_list)):
        if round(pix_width_list[0], prec) != round(pix_width_list[i], prec):
            print('\n\a Pixel widths of the input rasters are unqual, \
                        cannot continue to mosaic')
            return
    for i in range(0, len(pix_height_list)):
        if round(pix_height_list[0],prec) != round(pix_height_list[i], prec):
            print('\n\a Pixel heights of the input rasters are unqual, cannot \
                        continue to mosaic')
            return

    # Check to see if all the input rasters have the same input coordinate system,\
        #if not the function exits
    if same_coord_sys_ras_check(in_ras_list) is 1:
        print('\n\a Input coordinate systems of the input rasters do not match. \
                    Function exiting...')
        return

    # Getting the number of bands in a raster (the first one)
    # It is assumed that all the rasters have the same number of bands
    band_ras = gdal.Open(in_ras_list[0], 0)
    bands = band_ras.RasterCount

    # The output raster has the same driver as the input raster (first one)
    #driver = band_ras.GetDriver()
    driver = gdal.GetDriverByName(out_driver)

    # Assuming all the bands have the same datatype, resulting mosaic will have \
        # the datatype of the first band
    if out_data_type == 'input':
        GDAL_data_type = band_ras.GetRasterBand(in_band_no).DataType
    else:
        GDAL_data_type = out_data_type
        if GDAL_data_type is None:
            print('Could not get the GDAL datatype for the given out_data_type (%s). \
                    Exiting function' % out_data_type)
            return
        else:
            pass
    proj = band_ras.GetProjection() # Getting the projection
    if no_data_value=='input':
        try:
            # Getting the no_data_value of the first raster's first band, this is \
                # used as no_data_value for the all the bands of the mosaic (if no \
                # other value is specified)
            band = band_ras.GetRasterBand(in_band_no)
            in_NDV = band.GetNoDataValue()
        except:
            print('Cannot get no data value of the input raster. \
                        Please specify it. Exiting function...')
            return
    else:
        in_NDV = no_data_value
    band_ras = None

    # Creating the empty mosaiked raster
#        print 'Creating output raster %s' % out_ras
    out_cols = int(math.ceil((out_ras_x_max - out_ras_x_min) / pix_width_list[0]))
    out_rows = int(math.ceil((out_ras_y_max - out_ras_y_min) / abs(pix_height_list[0])))
    out_ras = driver.Create(out_ras, out_cols, out_rows, bands, GDAL_data_type, \
                            options=['COMPRESS='+comp_type])
#    if (mosaic_type=='minimum') or (mosaic_type=='maximum'):
    out_band = out_ras.GetRasterBand(in_band_no)

    GDT_in_bytes = GDT_to_bytes(GDAL_data_type)
    arr_size_in_bytes = out_cols * out_rows * GDT_in_bytes
    arr_size_in_MB = float(arr_size_in_bytes)/ (1000000) # convert array size to MB
    # setting the entire out_raster to null. Doing it in chunks because \
            # the array might be too big
    size_thresh = 1500. # array size in MB if an array is larger than this \
                            # then the array is read in slices
    if arr_size_in_MB > size_thresh:
        col_slice = int(arr_size_in_MB/size_thresh)
        row_slice = int(arr_size_in_MB/size_thresh)
        for l in range(0, (out_cols/col_slice)):
            if ((col_slice * l) + col_slice) > out_cols:
                #print 'if1 utilized'
                l_offset = (out_cols - (col_slice * l))
            else:
                l_offset = col_slice
            for m in range(0, (out_rows/row_slice)):
                if (row_slice * m) + row_slice > out_rows:
                    #print 'if2 utilized'
                    m_offset = (out_cols - (row_slice * m))
                else:
                    m_offset = row_slice

                if (l_offset > 0) and (m_offset > 0):
                    #print 'OutArr arguments are: ', out_cols, out_rows, (col_slice * l), (row_slice * m), l_offset, m_offset
                    out_arr = out_band.ReadAsArray((col_slice * l), (row_slice * m), l_offset, m_offset)
                    out_arr[:] = in_NDV
                    out_band.WriteArray(out_arr, (col_slice * l), (row_slice * m))
                else:
                    print("l or m Offset < 0 while filling mosaic with null values")
                    print('OutArr arguments are: ', out_cols, out_rows, (col_slice * l), (row_slice * m), l_offset, m_offset)
                    return
    else:
        out_arr = out_band.ReadAsArray(0, 0, out_cols, out_rows)
        out_arr[:] = in_NDV
        out_band.WriteArray(out_arr, 0, 0)
        out_band.FlushCache()
#    else:
##        print 'Output raster exists already.'
#        out_ras = gdal.Open(out_ras, 1)
#        out_band = out_ras.GetRasterBand(in_band_no)
#        out_cols = out_ras.RasterXSize
#        out_rows = out_ras.RasterYSize

    # Setting geotransform and projection
    out_ras.SetGeoTransform([out_ras_x_min, pix_width_list[0], 0, out_ras_y_max, 0, pix_height_list[0]])
    out_ras.SetProjection(proj)
#    print out_rows, out_cols
    # Defining the type of mosaic: last (default), first, average, blend, minimum, maximum
    in_ras_list2 = []
    if mosaic_type=='last':
        in_ras_list2=in_ras_list
    elif mosaic_type=='first':
        in_ras_list2=in_ras_list.reverse()
        x_min_list.reverse()
        x_max_list.reverse()
        y_min_list.reverse()
        y_max_list.reverse()
        rows_list.reverse()
        cols_list.reverse()
    elif mosaic_type=='average':
        #print 'average'
        in_ras_list2=in_ras_list
        # Creating a raster to count how many times a value is added to a pixel
        # This raster is divided, at the end, by output raster to get the average value of each pixel
        count_avg = driver.Create(temp_ras, out_cols, out_rows, bands, gdal.GDT_Byte, \
                                options=['COMPRESS='+comp_type])
        count_avg.SetGeoTransform([out_ras_x_min, pix_width_list[0], 0, out_ras_y_max, \
                                0, pix_height_list[0]])
        count_avg.SetProjection(proj)
    elif mosaic_type=='blend':
#        print 'blend'
        in_ras_list2=in_ras_list
    elif mosaic_type=='minimum':
#        print 'minimum'
        in_ras_list2=in_ras_list
    elif mosaic_type=='maximum':
#        print 'maximum'
        in_ras_list2=in_ras_list
    else:
        print('Incorrect mosaic-type specified. Exiting function')
        return

    # Going through each band of the input and output rasters and performing the required operations
    for k in range(1, int(bands)+1):
        index = 0
        out_band = out_ras.GetRasterBand(k)
        if (mosaic_type=='last') or (mosaic_type=='first'):
            # Writing values to mosaiked rasters
            for in_ras in in_ras_list2:
                in_ds = gdal.Open(in_ras, 0)
                in_band = in_ds.GetRasterBand(k)
                in_ndv = in_band.GetNoDataValue
                rows = rows_list[index]
                cols = cols_list[index]
                data_1 = in_band.ReadAsArray(0, 0, cols, rows)
#                print data_1.shape
                x_min = x_min_list[index]
                y_max = y_max_list[index]
                x_offset = int(math.floor((x_min - out_ras_x_min) / pix_width_list[0]))
                y_offset = int(math.floor((y_max - out_ras_y_max) / pix_height_list[0]))
#                print rows, cols, x_offset, y_offset
                data_2 = out_band.ReadAsArray(x_offset, y_offset, cols, rows)
#                print data_2.shape
                data_3 = np.where(data_1==in_ndv, data_2, data_1)
#                print data_3.shape

                out_band.WriteArray(data_3, x_offset, y_offset)
                index = index + 1
                in_ds = None

        if (mosaic_type=='maximum'):
            # Writing values to mosaiked rasters
            for in_ras in in_ras_list2:
                in_ds = gdal.Open(in_ras, 0)
                in_band = in_ds.GetRasterBand(k)
                in_ndv = in_band.GetNoDataValue
                rows = rows_list[index]
                cols = cols_list[index]
                data_1 = in_band.ReadAsArray(0, 0, cols, rows)
#                print data_1.shape
                x_min = x_min_list[index]
                y_max = y_max_list[index]
                x_offset = int(math.floor((x_min - out_ras_x_min) / pix_width_list[0]))
                y_offset = int(math.floor((y_max - out_ras_y_max) / pix_height_list[0]))
#                print rows, cols, x_offset, y_offset
                data_2 = out_band.ReadAsArray(x_offset, y_offset, cols, rows)
#                print data_2.shape
                data_3 = np.where(data_1==in_ndv, -np.Inf, data_1)
                data_4 = np.where(data_2==in_NDV, -np.Inf, data_2)
                data_5 = np.maximum(data_3, data_4)
                data_6 = np.where(data_5==-np.Inf, in_NDV, data_5)
#                print data_3.shape
                out_band.WriteArray(data_6, x_offset, y_offset)
                index = index + 1
                in_ds = None
        if (mosaic_type=='minimum'):
            # Writing values to mosaiked rasters
            for in_ras in in_ras_list2:
                in_ds = gdal.Open(in_ras, 0)
                in_band = in_ds.GetRasterBand(k)
                in_ndv = in_band.GetNoDataValue
                rows = rows_list[index]
                cols = cols_list[index]
                data_1 = in_band.ReadAsArray(0, 0, cols, rows)
#                print data_1.shape
                x_min = x_min_list[index]
                y_max = y_max_list[index]
                x_offset = int(math.floor((x_min - out_ras_x_min) / pix_width_list[0]))
                y_offset = int(math.floor((y_max - out_ras_y_max) / pix_height_list[0]))
#                print rows, cols, x_offset, y_offset
                data_2 = out_band.ReadAsArray(x_offset, y_offset, cols, rows)
#                print data_2.shape
                data_3 = np.where(data_1==in_ndv, np.Inf, data_1)
                data_4 = np.where(data_2==in_NDV, np.Inf, data_2)
                data_5 = np.minimum(data_3, data_4)
                data_6 = np.where(data_5==np.Inf, in_NDV, data_5)
#                print data_3.shape
                out_band.WriteArray(data_6, x_offset, y_offset)
                index = index + 1
                in_ds = None

        if (mosaic_type=='average'):
            # Writng a value of one to every pixel of the average count raster
            # Also adding the input raster to the empty mosaic raster such that overlapping values are added
            out_band_avg = count_avg.GetRasterBand(k)
            out_band_avg.SetNoDataValue(0)
            for i in range(0, len(in_ras_list)):
                x_min = x_min_list[i]
                y_max = y_max_list[i]
                rows = rows_list[i]
                cols = cols_list[i]
                x_offset = int(math.floor((x_min - out_ras_x_min) / pix_width_list[0]))
                y_offset = int(math.floor((y_max - out_ras_y_max) / pix_height_list[0]))
                in_ds = gdal.Open(in_ras_list[i], 0)
                in_band = in_ds.GetRasterBand(k)
                # For adding all the raster to the main raster
                data_4 = out_band.ReadAsArray(x_offset, y_offset, cols, rows)
                in_data = in_band.ReadAsArray(0,0, cols, rows)
                new_data = np.add(data_4, in_data)
                out_band.WriteArray(new_data, x_offset, y_offset)
                # For writing data to the overlap-count raster
                ones_data = np.ones((rows, cols), dtype=np.int)
                ones_data[in_data==in_NDV] = 0
                pre_array = out_band_avg.ReadAsArray(x_offset, y_offset, cols, rows)
                new_array = np.add(ones_data, pre_array)
                out_band_avg.WriteArray(new_array, x_offset, y_offset)
                in_ds = None
            out_band_avg.FlushCache()
            out_band.FlushCache()

        try:
            if no_data_value=='input':
                NDV = in_NDV
            else:
                NDV = no_data_value
        except:
            print('Incorrect no-data-value supplied. Please check and then run again. Function exiting...')
            return

        if (mosaic_type=='average'):
            # Averaging
            data_5 = out_band.ReadAsArray(0, 0, out_cols, out_rows)
            data_6 = out_band_avg.ReadAsArray(0, 0, out_cols, out_rows)
            average = np.divide(data_5, data_6)
            average[data_6 == 0] = in_NDV
            #average[average <= 0] = in_NDV #  Depends how NDV occurs in overlapping cells
            out_band.WriteArray(average, 0, 0)
            out_band.FlushCache()
            count_avg = None
            # Deleting the temporary raster
            os.remove(temp_ras)
        if (any_NDV_to_NDV is True):
            for i in range(0, len(in_ras_list)):
                x_min = x_min_list[i]
                y_max = y_max_list[i]
                rows = rows_list[i]
                cols = cols_list[i]
                x_offset = int(math.floor((x_min - out_ras_x_min) / pix_width_list[0]))
                y_offset = int(math.floor((y_max - out_ras_y_max) / pix_height_list[0]))
                in_ds = gdal.Open(in_ras_list[i], 0)
                in_band = in_ds.GetRasterBand(k)
                # For adding all the raster to the main raster
                data_7 = out_band.ReadAsArray(x_offset, y_offset, cols, rows)
                in_data = in_band.ReadAsArray(0, 0, cols, rows)
                data_7[in_data==in_NDV] = in_NDV
                out_band.WriteArray(data_7, x_offset, y_offset)
                in_ds = None
        out_band.SetNoDataValue(NDV)
        out_band.FlushCache()

    out_ras = None
    return 0


# Function to create plots using MatplotLib and some other modules (A legend with colors representing each value palced on the right side)
    #obsolete, use the one from IS plots or chitral plots
def drawMapFunc_discrete(inRas, inShp, outFig, outProj, colorList, labelList, \
            rasValList, legendTitle, outdpi):
    # Getting extents of the intput raster
    ext = get_ras_props(inRas)

    # Specifying the projection and extents of the plot area, UTM is not supported using Basemap so we just use regular coordinates
    m = Basemap(projection=outProj, llcrnrlat=(ext[2]),urcrnrlat=(ext[3]),llcrnrlon=(ext[0]),urcrnrlon=(ext[1]))

    # Specifying the shapefile to plot on the map as well, can be null
    m.readshapefile(inShp[:-4], 'dntknwwat')

    # Reading the input raster as an array and flipping it upside down ( has to be done )
    ds = gdal.Open(inRas)
    data = ds.ReadAsArray()
    data = np.flipud(data)
    data = np.ma.masked_where(data < min(rasValList), data)
    data = np.ma.masked_where(data > max(rasValList), data)

    # Color list converted to a listed color map
    cMap = ListedColormap(colorList)
    normL = mpl.colors.BoundaryNorm(boundaries=rasValList, ncolors=len(colorList))
    proxy = [plt.Rectangle((0,0),1,1,fc = pc) for pc in colorList]

    # Latitude's and longitude's increments
    latIncr = (ext[3] - ext[2])/3
    lonIncr = (ext[1] - ext[0])/3
    parallels = np.around(np.arange(ext[2],ext[3],latIncr), decimals=2)
    meridians = np.around(np.arange(ext[0],ext[1],lonIncr), decimals=2)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[1,0,0,0], fontsize=6) # draw parallels
    m.drawmeridians(meridians,labels=[0,0,0,1], fontsize=6) # draw meridians

    # convert array to mesh
    x = linspace(0, m.urcrnrx, data.shape[1])
    y = linspace(0, m.urcrnry, data.shape[0])
    xx, yy = meshgrid(x, y)
    plt.ioff()
    plt.pcolormesh(xx,yy,data,cmap=cMap, norm=normL)

    # Draw map legend and scale
    plt.legend(proxy, labelList, bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0., fancybox=1, title=legendTitle, mode=None, prop={'size':8})
    m.drawmapscale((ext[0] + ext[1])/2, (ext[2] + (0.07*(ext[3]-ext[2]))), ext[0], ext[2], 100, barstyle='fancy', units='km', \
            fontsize=9, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', \
            fillcolor2='k', ax=None, format='%d', zorder=None)

    plt.savefig(outFig, dpi=outdpi, edgecolor='black', bbox_inches='tight', fontsize=8)
    plt.close('all')
    ds = None
    return

# Function to create plots using MatplotLib and someother modules
    #obsolete, use the one from IS plots
def drawMapFunc_continuous(inRas, inShp, outFig, proj, rasMin, rasMax, colorBarPos, \
            ColorBarName, colorBarTickInterval, outdpi, outFormat):
    ext = get_ras_props(inRas)
    m = Basemap(projection=proj, llcrnrlat=(ext[2]),urcrnrlat=(ext[3]),llcrnrlon=(ext[0]),urcrnrlon=(ext[1]))
    m.readshapefile(inShp[:-4], 'dntknwwat')
    ds = gdal.Open(inRas)
    data = ds.ReadAsArray()
    data = np.flipud( data )
    data = np.ma.masked_greater(data, rasMax)
    data = np.ma.masked_less(data, rasMin)
    latIncr = (ext[3] - ext[2])/3
    lonIncr = (ext[1] - ext[0])/3
    parallels = np.around(np.arange(ext[2],ext[3],latIncr), decimals=2)
    meridians = np.around(np.arange(ext[0],ext[1],lonIncr), decimals=2)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(meridians,labels=[0,0,0,1]) # draw meridians
    x = linspace(0, m.urcrnrx, data.shape[1])
    y = linspace(0, m.urcrnry, data.shape[0])
    xx, yy = meshgrid(x, y)
    colormesh = m.pcolormesh(xx, yy, data)
    cb = m.colorbar(colormesh, location=colorBarPos, label=ColorBarName)
    m.drawmapscale((ext[0] + ext[1])/2, (ext[2] + (0.07*(ext[3]-ext[2]))), ext[0], ext[2], 100, barstyle='fancy', units='km', \
            fontsize=9, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', \
            fillcolor2='k', ax=None, format='%d', zorder=None)
    cb.set_ticks(list(range(rasMin, rasMax, colorBarTickInterval)))
    plt.ioff()
    plt.savefig(outFig, dpi=outdpi, edgecolor='black')
    plt.close('all')
    ds = None
    return


def mask_ras_with_shp_GDALWARP(in_ras, out_ras,
                               mask_shp,
                               in_band_no=1,
                               no_data_value='input',
                               warp_file=warp_file,
                               res_meth='near',
                               out_driver='GTiff',
                               comp_type='LZW'):
    """
    Purpose: To mask a raster using a given layer in a shapefile.

    Description of the arguments:

        in_ras (string): Full path to the input raster.

        out_ras(string): Full path of the output raster.

        mask_shp (string): Full path of the shape file used as a mask. \
                It masks for all the polygons in the shapefile. Extents \
                of out_ras depend on the extents of the shapefile.

        in_band_no (int): Band number of the raster to use for masking.

        no_data_value (int or float): NoData value of the output raster. \
                if 'input' then the value on the input raster is used.

        warp_file (string): Full path to the GDALWARP executable. Specified in the \
                begining of this script.

        res_meth (string): The resampling method to use while reprojecting. \
                    Can be 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', \
                    'average', 'mode'. Defaults to 'near' which requires the least \
                    computational effort with worst accuracy for continuous data. \
                    It is good for categorical data though.

        Note: May change output raster size.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds == None:
        print('Could not open input raster (%s)' % in_ras)
        return
    else:
        in_ds = None
        props = list(map(str, get_ras_props(in_ras, in_band_no=1)))
        if no_data_value == 'input':
                NDV = props[9]
        else:
            NDV = str(no_data_value)
            #(' -te %s %s %s %s -tr %s %s -tap' %(props[0], props[2], props[1], props[3], props[6], props[7])) + \
    arg = warp_file.replace("/", "\\"), (' -srcnodata %s  -dstnodata %s -r %s' %(NDV, NDV, res_meth)) + ' -cutline ' + \
            os.path.dirname(mask_shp).replace("/", "\\") + ' -cl ' + os.path.basename(mask_shp).rsplit('.',1)[0] + ' -crop_to_cutline' + \
            (' -te %s %s %s %s -tr %s %s -tap' %(props[0], props[2], props[1], props[3], props[6], props[7])) + \
             " -of " + out_driver + " -co COMPRESS=" + comp_type + \
            ' -overwrite -q -wo SKIP_NOSOURCE=YES' + " " + in_ras.replace("/", "\\") + " " + out_ras.replace("/", "\\")
    subprocess.call([arg])
    return 0

def resample_ras_GDALWARP(in_ras, out_ras, tr_xres, tr_yres, res_meth='near', in_band_no=1, \
                          no_data_value='input', warp_file=warp_file, comp_type='LZW', out_driver='GTiff'):
    """
    Purpose: To resample a raster's cell size.
    Description of the arguments:
        in_ras (string): Full path to the input raster.
        out_ras (string): Full path of the output raster.
        tr_xres (int or float): cell size in x-direction.
        tr_yres (int or float): cell size in y-direction.
        res_meth (string): The resampling method to use. \
                    Can be 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', \
                    'average', 'mode'. Defaults to 'near' which require the least \
                    computational effort with worst accuracy for continuous data. \
                    It is good for categorical data though.
        in_band_no (int): Band number of the raster to use for masking.
        no_data_value (int or float): NoData value of the output raster. \
                if 'input' then the value on the input raster is used.
        warp_file (string): Full path to the GDALWARP executable. Specified in the \
                begining of this script.

    """
    in_ds = gdal.Open(in_ras, 0)
    if no_data_value == None:
        print('Could not open input raster (%s)')
        return
    else:
        if no_data_value == 'input':
            try:
                NDV = in_ds.GetRasterBand(in_band_no).GetNoDataValue()
            except:
                print('Could not get the noData value from the input raster. Specify manually!')
                return
        else:
            NDV = no_data_value
        in_ds = None
    arg = warp_file.replace("/", "\\"), ' -srcnodata ' + str(NDV) + ' -dstnodata ' + str(NDV) + " -co COMPRESS=" + comp_type + \
            ' -tr ' + str(tr_xres) + " " + str(tr_yres) + ' -r ' + res_meth  + " -of " + out_driver + ' ' + \
            in_ras.replace("/", "\\") + ' -overwrite -q -tap ' + out_ras.replace("/", "\\")
    subprocess.call([arg])
    return

def same_coord_sys_ras_check(in_ras_list):
    """
    Purpose: To check if the specified input rasters have the same coordinate system. \
                If not the function returns 1.
    Description of the arguments:
        in_ras_list (List of string): List of the input rasters containing full \
                paths to the input rasters.
    """
    ras_0 = gdal.Open(in_ras_list[0], 0)
    proj_0 = ras_0.GetProjection()
    for h in range (1, len(in_ras_list)):
        ras_h = gdal.Open(in_ras_list[h], 0)
        if (ras_0 == None) or (ras_h == None):
            print('\n\a Could not open one of the input rasters. Exiting function...')
            ras_0 = ras_h = None
            return
        else:
            proj_h = ras_h.GetProjection()
            if proj_0 != proj_h:
                print('\n\a Input rasters (%s and %s) do not have the same coordinate system.' \
                            % (in_ras_list[0], in_ras_list[h]))
                ras_0 = ras_h = None
                return
            else:
#                print '\n\a Input rasters (%s and %s) have the same coordinate system.'  \
#                            % (in_ras_list[0], in_ras_list[h])
                pass
    ras_0 = ras_h = None
    return 0

def Set_null(in_ras, out_ras, low_lim, up_lim, in_band_no=1, \
                no_data_value='input', comp_type='LZW', out_driver='GTiff'): # untested
    """
    Purpose: To set values to noData a raster given upper and lower limits. \
                The interval is close (> or <).
    Description of arguments:
        in_ras (string): Full path to the input raster.
        out_ras (string): Full path of the output raster. This raster will \
                    have those values set to null in the input raster which are \
                    outside the given interval.
        low_lim (int or float): The lower limit of values in the raster below which values \
                    are turned to null (inclusive).
        up_lim (int or float): The upper limit of values in the raster above which \
                    values are set to null (inclusive).
        in_band_no (int): The band number to use (starts from 1)
        no_data_value (int or float): noData value of the output raster. By default the \
                    value of input raster is used.
    """

    inDs = gdal.Open(in_ras, 0)
    if inDs == None:
        print('\n\a Could not open input raster (%s)' % in_ras)
        return
    else:
        try:
            inBand = inDs.GetRasterBand(in_band_no)
            NDV = inBand.GetNoDataValue
            GDALDataType = inBand.DataType
            rows = inDs.RasterYSize
            cols = inDs.RasterXSize
            data = inBand.ReadAsArray(0,0, rows, cols)
            data[data<low_lim] = NDV
            data[data>up_lim] = NDV
            driver = gdal.GetDriverByName(out_driver)
            outDs = driver.Create(out_ras, rows, cols, 1, GDALDataType, \
                                options=['COMPRESS='+comp_type])
            outBand = outDs.GetRasterBand(1)
            outBand.WriteArray(data, 0, 0)
            if no_data_value == 'input':
                try:
                    NDV = inBand.GetNoDataValue()
                except:
                    print('\n\a Could not get the noData value from the input raster. Specify manually!')
                    return
            else:
                NDV = no_data_value
            outBand.SetNoDataValue(NDV)
            outBand.FlushCache()
            geotransform = inDs.GetGeoTransform()
            outDs.SetGeoTransform(geotransform)
            proj = inDs.GetProjection()
            outDs.SetProjection(proj)
            outDs = None
            inDs = None
        except:
            print('\n\a Something went wrong while setting values to null. \
                        Check the intput data.')
            return
    return 0

def new_ras_GDAL(out_ras, x_l_c, y_u_c, cols, rows, x_cell_size, y_cell_size, \
                coord_sys, band_count=1, out_driver='GTiff', \
                out_data_type=gdal.GDT_Float32, comp_type='LZW'):
    """
    Purpose: To create a new raster given the required parameters
    Description of the arguments:
        out_ras (string): Full path to the output raster. The driver to create \
                        the outRas is specified with the 'driver' argument.
        x_l_c (int or float): Left side coordinate in x-axis (x-minimum).
        y_u_c(int or float): Upper side coordinate in y-axis (y-maximum).
        cols (int): Number of columns the raster should have.
        rows (int): Number of rows the raster should have.
        x_cell_size (int or float, positive): Width of the cells in x-direction.
        y_cell_size (int or float, positive): Height of the cells in y-direction.
        coord_sys (.prj file or the projection of a raster): A .prj file having the \
                        required coordinate system. Can be downloaded \
                        from spatialreference.org.
        band_count (int): Required no. of bands in the outRas.
        out_driver (string, short format name): Short name of the file system \
                        according to GDAL. Defaults to GTiff.
        out_data_type (GDAL datatype): Corresponds to GDAL data type. Sometimes, while \
                        running the function an error may occur saying that \
                        there is no module named GDAL, in that case use the \
                        'str_to_GDT' function.
    """
    out_driver = gdal.GetDriverByName(out_driver)
    new_ras = out_driver.Create(out_ras, cols, rows, band_count, \
                out_data_type, options=['COMPRESS='+comp_type])
    new_ras.SetGeoTransform([x_l_c, x_cell_size, 0, y_u_c, 0, -y_cell_size])
    proj = osr.GetUserInputAsWKT(coord_sys)
    new_ras.SetProjection(proj)
    new_ras = None
    return 0

def new_ras_from_ASCII_GDAL(out_ras, ASCII_file, coord_sys, out_driver='GTiff', \
                        out_data_type=gdal.GDT_Float32, comp_type='LZW'):
    """
    Purpose: To create a new raster using an ASCII format text file.
    Description of the arguments:
        out_ras (string): Full path of the output raster. The driver to create \
                    the outRas is specified with the 'driver' argument. \
                    It's deleted if it exists already.
        ASCII_file (string): Full path to the ASCII text file. All the variables \
                    are read from this file. Take a look at how everything compares \
                    with things defined here. Maybe the order is not the same.
        coord_sys (.prj file or projection of a raster): A .prj file having the \
                    required coordinate system. Can be downloaded from spatialreference.org.
        out_driver (string): Short format name of the file system according to GDAL.
        out_data_type (GDAL datatype): Corresponds to GDAL data type. Sometimes, while running the \
                    function an error may occur saying that there is no module named GDAL.
                    so maybe we need to import it, although it is imported in faizpy already.\
                    Alternatively we can use the 'str_to_GDT' function.
    """

    cols = int(linecache.getline(ASCII_file, 1).split()[1])
    rows = int(linecache.getline(ASCII_file, 2).split()[1])
    x_l_c = float(linecache.getline(ASCII_file, 3).split()[1])
    cell_size = float(linecache.getline(ASCII_file, 5).split()[1])
    no_data_value = float(linecache.getline(ASCII_file, 6).split()[1])
    y_u_c = float(linecache.getline(ASCII_file, 4).split()[1]) + float(rows*cell_size)

    if os.path.exists(out_ras):
        os.remove(out_ras)

    out_driver = gdal.GetDriverByName(out_driver)
    new_ras = out_driver.Create(out_ras, cols, rows, 1, out_data_type, options=['COMPRESS='+comp_type])
    new_ras.SetGeoTransform([x_l_c, cell_size, 0, y_u_c, 0, -cell_size])

    proj = osr.GetUserInputAsWKT(coord_sys)
    new_ras.SetProjection(proj)

    band = new_ras.GetRasterBand(1)
    data = np.loadtxt(ASCII_file, skiprows=6)
    band.WriteArray(data, 0, 0)
    band.FlushCache()
    band.SetNoDataValue(no_data_value)
    new_ras = None
    return None

def ras_to_ASCII_GDAL(in_ras, out_ASCII_file, in_band_no=1, prec=1):
    """
    Purpose: To write a raster, with the required parameters, to an ASCII text file.
    Description of arguments:
        in_ras (string): Full path to the input raster.
        out_ASCII_file (string): Full path to the output ASCIIFile. If the file \
                        exists already, it's deleted.
        in_band_no (int): The band number of the input used for writing to the ASCII file.
        prec (int): The decimal point precision of the output values.
    """

    in_ds = gdal.Open(in_ras, 0)
    in_band = in_ds.GetRasterBand(in_band_no)
    NDV = in_band.GetNoDataValue()
    rows = in_ds.RasterYSize
    cols = in_ds.RasterXSize
    in_arr = in_band.ReadAsArray(0, 0, cols, rows)
    geotransform = in_ds.GetGeoTransform()
    x_min = geotransform[0]
    y_max = geotransform[3]
    pix_width = geotransform[1]
    pix_height = abs(geotransform[5])
    #xMax = xMin + (cols*pixWidth)
    y_min = y_max - (rows*pix_height)
    #extent = [xMin, xMax, yMin, yMax, cols, rows, pixWidth, pixHeight]
    #print extent
    if round(pix_width, 5) != round(pix_height, 5):
        print('Pixels of the input raster are not square. Function exiting...')
        return
    if os.path.exists(out_ASCII_file):
        os.remove(out_ASCII_file)

    cursor = open(out_ASCII_file, 'w')
    cursor.write('ncols\t' + str(cols) + '\n')
    cursor.write('nrows\t' + str(rows) + '\n')
    cursor.write('xllcorner\t' + str(x_min) + '\n')
    cursor.write('yllcorner\t' + str(y_min) + '\n')
    cursor.write('cellsize\t' + str(pix_width) + '\n')
    cursor.write('NODATA_value\t' + str(NDV)+ '\n')
    np.savetxt(cursor, in_arr, fmt='%1.' + str(prec) + 'f')
    cursor.close()
    in_ds = None
    return 0

def same_extents_ras_check(in_ras_list, prec=6):
    """
    Purpose: To check if the horizontal/vertical extents, cell size, number \
                of rows/columns of a given set of rasters is same.
    Description of the arguments:
        in_ras_list (list of strings): List of the input raster having full path \
                to the rasters that need to be compared.
        prec: The decimal point precision of numbers for comparison of the \
                rasters' dimensions.
    """
    ext_0 = get_ras_props(in_ras_list[0])
    for h in range (1, len(in_ras_list)):
        ext_h = get_ras_props(in_ras_list[h])
        if (round(ext_0[0], prec))!=(round(ext_h[0], prec)) or (round(ext_0[1], \
                        prec))!=(round(ext_h[1], prec)):
            print('\n\a Rasters (%s and %s) do not have same extents in horizontal direction.'\
                    % (in_ras_list[0], in_ras_list[h]))
            return
        elif (round(ext_0[2], prec))!=(round(ext_h[2], prec)) or (round(ext_0[3], \
                        prec))!=(round(ext_h[3], prec)):
            print('\n\a Rasters (%s and %s) do not have same extents in vertical direction.'\
                    % (in_ras_list[0], in_ras_list[h]))
            return
        elif (round(ext_0[4], prec))!=(round(ext_h[4], prec)) or (round(ext_0[5], \
                        prec))!=(round(ext_h[5], prec)):
            print('\n\a Rasters (%s and %s) do not have the same number of columns or rows.'\
                    % (in_ras_list[0], in_ras_list[h]))
            return
        elif (round(ext_0[6], prec))!=(round(ext_h[6], prec)) or (round(ext_0[7], \
                        prec))!=(round(ext_h[7], prec)):
            print('\n\a Rasters (%s and %s) do not have same horizontal or vertical cell size.'\
                    % (in_ras_list[0], in_ras_list[h]))
            return
        else:
            pass
#    print 'All rasters have the same extents'
    return 0

def get_ras_NDV(in_ras, in_band_no=1):
    """
    Purpose: To get the nodata value of a given raster and band.
    Description of the arguments:
        in_ras (string): Full path to the input raster.
        in_band_no (int): The band number for which  we need the noData value.
    """
    in_ds = gdal.Open(in_ras, 0)
    in_band = in_ds.GetRasterBand(in_band_no)
    try:
        NDV = in_band.GetNoDataValue()
        in_ds = None
        return NDV
    except:
        print('Could not get the noData value of %s' % in_ras)
        in_ds = None
        return

def get_ras_GDAL_data_type(in_ras, in_band_no=1):
    """
    Purpose: To get the GDAL datatype of the given band of a raster.
    Description of the arguments:
        in_ras (string): Full path of the input raster.
        in_band_no (int): The band number of which GDAL datatype is needed.
    """
    in_ds = gdal.Open(in_ras, 0)
    dt = in_ds.GetRasterBand(in_band_no).DataType
    in_ds = None
    return dt

def get_ras_as_array_GDAL(in_ras, in_band_no=1):
    """
    Purpose: To return the given raster's band as an array.
    Description of arguments:
        inRas (string): Full path to the input raster.
        in_band_no (int): The band number which has to be read.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds == None:
        print('Could not open %s for reading' % in_ras)
        return
    else:
        in_band = in_ds.GetRasterBand(in_band_no)
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize
        ras_arr = np.array(in_band.ReadAsArray(0, 0, cols, rows))
        in_ds = None
        return np.flipud(ras_arr)

def get_ras_coord_sys_as_WKT(in_ras):
    """
    Purpose: To get the coordinate system of a raster in Well-Known-Text \
                (WKT) format.
    Description of the arguments:
        in_ras (string): Full path of the input raster.
    """
    in_ds = gdal.Open(in_ras, 0)
    proj = in_ds.GetProjection()
    in_ds = None
    return osr.GetUserInputAsWKT(proj)

#def subplot_bar_dsctMap(inRas, outFig, outTxtFile, colorList, labelList, \
#            rasValList, legendTitle, comment='', fs=10, outdpi=300, res='i', \
#            inShp='c:\NoShp.shp', pars=3, mers=3):
#    # Function to create plots using MatplotLib and some other modules \
#            # (A legend with colors representing each value palced on the right side)
#
#    f , ax  = plt.subplots(1, 2, figsize=(23,10))
#
#    # Reading the input raster as an array and flipping it upside down ( has to be done )
#    ds = gdal.Open(inRas)
#    data = ds.ReadAsArray()
#    data = np.flipud(data)
#    data = np.ma.masked_where(data < min(rasValList), data)
#    data = np.ma.masked_where(data > max(rasValList), data)
#    ds = None
#
#    plt.subplot(1, 2, 1)
#    footerStr='\n' + os.path.basename(inRas) + '\t' + legendTitle + '\t' + os.path.basename(inShp) + '\t'
#    for i in range(0, len(rasValList)):
#        height = data[data==rasValList[i]].size
#        plt.bar(i, height, width=0.8, color=colorList[i], align='center')
#        footerStr += str(height) + '\t'
#    stats = rasterstats.zonal_stats(inShp, inRas, stats=['count'])
#    count = stats[0].items()[0][1]
#    footerStr += str(count)
#    footer = open(outTxtFile, 'a')
#    #print footerStr
#    footer.write(footerStr)
#    footer.close()
#
##    plt.title(legendTitle + ' Bar Chart (Date: ' + os.path.basename(inRas)[:-4] + ')')
#    plt.xlabel('Surface Type')
#    plt.ylabel('Cell Count')
#    plt.xticks(range(0, len(rasValList)), labelList, fontsize=fs, rotation=90)
#    plt.subplots_adjust(bottom=0.15)
#    #plt.margins(0.2)
##    axes = plt.gca()
##    axes.set_xticklabels(labelList, fontsize=fs, verticalalignment='center')
#
#    plt.subplot(1, 2, 2)
#    # Getting extents of the intput raster
#    ext = get_ras_props(inRas)
#
#    # Specifying the projection and extents of the plot area, UTM is not supported using Basemap so we just use regular coordinates
#    m = Basemap(projection='merc', llcrnrlat=(ext[2]),urcrnrlat=(ext[3]),llcrnrlon=(ext[0]),urcrnrlon=(ext[1]), resolution=res)
#
#    # Specifying the shapefile to plot on the map as well, can be null
#    if os.path.exists(inShp):
#        m.readshapefile(inShp[:-4], 'dntknw')
#    else:
#        print 'No shapefile'
#    #print inShp
#    # Color list converted to a listed color map
#    cMap = ListedColormap(colorList)
#    normL = mpl.colors.BoundaryNorm(boundaries=rasValList, ncolors=len(colorList))
#    proxy = [plt.Rectangle((0,0),1,1,fc = pc) for pc in colorList]
#
#    # Latitude's and longitude's increments
#    latIncr = (ext[3] - ext[2])/pars
#    lonIncr = (ext[1] - ext[0])/mers
#    parallels = np.around(np.arange(ext[2],ext[3],latIncr), decimals=2)
#    meridians = np.around(np.arange(ext[0],ext[1],lonIncr), decimals=2)
#    # labels = [left,right,top,bottom]
#    m.drawparallels(parallels,labels=[1,0,0,0], fontsize=fs) # draw parallels
#    m.drawmeridians(meridians,labels=[0,0,0,1], fontsize=fs) # draw meridians
#
#    # convert array to mesh
#    x = linspace(0, m.urcrnrx, data.shape[1]+1)
#    y = linspace(0, m.urcrnry, data.shape[0]+1)
#    xx, yy = meshgrid(x, y)
#    plt.ioff()
#    plt.pcolormesh(xx,yy,data,cmap=cMap, norm=normL)
#
#    # Draw map legend and scale
#    plt.legend(proxy, labelList, bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0., fancybox=1, title=legendTitle, mode=None, prop={'size':fs})
#
#    m.drawmapscale((ext[0] + ext[1])/2, (ext[2] + (0.07*(ext[3]-ext[2]))), ext[0], ext[2], 100, barstyle='fancy', units='km', \
#            fontsize=fs, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', \
#            fillcolor2='k', ax=None, format='%d', zorder=None)
#
#    f.text(0.1, -.01, comment, verticalalignment='top')
#    plt.margins(0.2)
##    plt.tight_layout()
#
#    plt.savefig(outFig, dpi=outdpi, edgecolor='black', bbox_inches='tight', fontsize=fs)
#    plt.close('all')
#    return
#
#def subplot_hist_contMap(inRas, outFig, outTxtFile, minVal, maxVal, cBarName, \
#            cBarTickIntvl, histBins, normed=0, cumulative=0, comment='', \
#            cBarPos='right', outdpi=300, res='i',inShp='c:\\inshp.shp', pars=3, \
#            mers=3, dsName=''):
#    f , ax  = plt.subplots(2, 2, figsize=(23,10))
#    #matplotlib.figure.SubplotParams(wspace=.50, hspace=.50, left  = 0.125, \
#                #right = 0.9, bottom = 0.1, top = 0.9)
#
#    ds = gdal.Open(inRas)
#    data = ds.ReadAsArray()
#    ds = None
#    data = np.array(data, dtype=np.float32)
#    data = np.ma.masked_greater(data, maxVal)
#    data = np.ma.masked_less(data, minVal)
#    data2 = np.transpose(data)
#
#    plt.subplot(1, 2, 1)
#    plt.hist(data2, range=(minVal, maxVal), bins=histBins, histtype='bar', orientation='vertical', align='mid', normed=normed, stacked=True, cumulative=cumulative)
#    hist, bin_edges = np.histogram(data2, bins=histBins, range=(minVal, maxVal), )
#    footerStr='\n' + os.path.basename(inRas) + '\t' + dsName + '\t' + os.path.basename(inShp) + '\t'
#    for i in range(0, len(hist)):
#        #print 'Bins(', bin_edges[i], '-', bin_edges[i+1], '): ', hist[i]
#        footerStr += str(hist[i]) + '\t'
#    stats = rasterstats.zonal_stats(inShp, inRas, stats="count min max median range mean")
#    count = str(stats[0].items()[0][1])
#    mini = str(stats[0].items()[1][1])
#    maxi = str(stats[0].items()[2][1])
#    median = str(stats[0].items()[3][1])
#    rangei = str(stats[0].items()[4][1])
#    mean = str(stats[0].items()[6][1])
#    statList = [count, mini, maxi, median, rangei, mean]
#    for i in range(len(statList)):
#        footerStr += statList[i] + '\t'
#    footer = open(outTxtFile, 'a')
#    #print footerStr
#    footer.write(footerStr)
#    footer.close()
##    plt.title(cBarName + ' Histogram (Date: ' + os.path.basename(inRas)[:-4] + ')')
#    plt.xlabel('Value (%)')
#    plt.ylabel('Cell Count')
#    axes = plt.gca()
#    axes.set_xticks(range(minVal, maxVal+1, histBins))
#
#    plt.subplot(1, 2, 2)
#    ext = get_ras_props(inRas)
#    m = Basemap(projection='merc', llcrnrlat=(ext[2]),urcrnrlat=(ext[3]),llcrnrlon=(ext[0]),urcrnrlon=(ext[1]), resolution=res)
#    if os.path.exists(inShp):
#        m.readshapefile(inShp[:-4], 'dntknw')
#    else:
#        print 'No shapefile'
#    latIncr = (ext[3] - ext[2])/pars
#    lonIncr = (ext[1] - ext[0])/mers
#    parallels = np.around(np.arange(ext[2],ext[3],latIncr), decimals=2)
#    meridians = np.around(np.arange(ext[0],ext[1],lonIncr), decimals=2)
#    # labels = [left,right,top,bottom]
#    data = np.flipud(data)
#    m.drawparallels(parallels,labels=[1,0,0,0]) # draw parallels
#    m.drawmeridians(meridians,labels=[0,0,0,1]) # draw meridians
#    x = linspace(0, m.urcrnrx, data.shape[1]+1)
#    y = linspace(0, m.urcrnry, data.shape[0]+1)
#    xx, yy = meshgrid(x, y)
#    colormesh = m.pcolormesh(xx, yy, data)
#    cb = m.colorbar(colormesh, location=cBarPos, label=cBarName)
#    m.drawmapscale((ext[0] + ext[1])/2, (ext[2] + (0.07*(ext[3]-ext[2]))), ext[0], ext[2], 100, barstyle='fancy', units='km', \
#            fontsize=9, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', \
#            fillcolor2='k', ax=None, format='%d', zorder=None)
#    cb.set_ticks(range(minVal, maxVal+1, cBarTickIntvl))
#
#    # To add a text
##    f.text(0.1, -.01, comment + '\nFull file path: ' + outFig , verticalalignment='top')
#    f.text(0.1, -.01, comment, verticalalignment='top')
##    plt.tight_layout()
#    plt.ioff()
#    plt.savefig(outFig, dpi=outdpi, edgecolor='black', bbox_inches='tight')
#    plt.close('all')
#    return

def change_ras_GDT(in_ras, out_ras, in_band_no=1, out_driver ='input', no_data_value = 'input', \
                new_data_type=gdal.GDT_Float32, comp_type='LZW'):
    """
    Purpose: To change the GDAL dataType of a raster.
    Description of the arguments:
        in_ras (string): Full path to the input raster
        out_ras (string): Full path to the output raster. It's deleted, if it \
                            exists already.
        out_driver (string): Short format name of the output driver. \
                            Defaults to the driver of the input raster.
        no_data_value (int or float): noData value of the output raster. \
                            Defaults to that of the input raster.
        new_data_type (GDAL datatype): The new GDAL datatype that we need for the \
                            output raster.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds == None:
        print('\n\a Could not open input raster (%s)' % in_ras)
        return
    else:
        if no_data_value == 'input':
            try:
                in_band = in_ds.GetRasterBand(in_band_no)
                NDV = in_band.GetNoDataValue()
            except:
                print('\n\a Could not get the noData value from the input raster (%s).'\
                        'Specify it manually!' % in_ras)
                return
        else:
            NDV = no_data_value

        if out_driver == 'input':
            out_driver = in_ds.GetDriver().ShortName

    geotransform = in_ds.GetGeoTransform()
    rows = in_ds.RasterYSize
    cols = in_ds.RasterXSize
    band_count = in_ds.RasterCount
    in_arr = in_band.ReadAsArray(0,0, cols, rows)
    x_min = geotransform[0]
    y_max = geotransform[3]
    pix_width = geotransform[1]
    pix_height = abs(geotransform[5])
    proj = in_ds.GetProjection()
    coord_sys = osr.GetUserInputAsWKT(proj)

    if os.path.exists(out_ras):
        os.remove(out_ras)

    new_ras_GDAL(out_ras, x_min, y_max, cols, rows, pix_width, pix_height, coord_sys, \
                band_count=band_count, out_driver=out_driver, \
                out_data_type=new_data_type, comp_type=comp_type)
    out_ds = gdal.Open(out_ras, 1)
    out_band = out_ds.GetRasterBand(in_band_no)
    out_arr = in_arr.astype(GDT_to_NPDT(new_data_type))
    out_band.WriteArray(out_arr, 0, 0)
    out_band.SetNoDataValue(NDV) # setting NDV needs work
    out_band.FlushCache()
    out_ds = None
    in_ds = None
    return 0


def chng_ras_fmt_GDAL_Translate(in_ras, out_ras, in_band_no=1, out_driver='GTiff', \
    out_data_type='Int16', no_data_value='input', comp_type='LZW', translate_file=translate_file):
    """
    Purpose: To change the datatype of a raster band and write it to a new file \
                using GDAL TRANSLATE.
    Description of the arguments:
        in_ras (string): Full path to the input raster.
        out_ras (string): Full path of the ouput raster. Overwritten if exists already.
        in_band_no (int): The band number of the input raster which we need to change.
        out_driver (string): Short format name of the output raster's format
        out_data_type (string): output data type name. It can be 'Byte', 'Int16', \
                        'UInt16', 'Int32', 'UInt32', 'Float32', 'Float64'.
        no_data_value (int or float): noData value of the output raster raster. \
                        Can be None.
        translate_file (string): Full path to the GDAL Translate executable.

    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds == None:
        print('\n\a Could not open input raster (%s)' % in_ds)
        return
    else:
        if no_data_value == 'input':
            try:
                NDV = in_ds.GetRasterBand(in_band_no).GetNoDataValue()
            except:
                print('\n\a Could not get the noData value of the input raster (%s).' \
                ' Specify manually!' % in_ras)
                return
        else:
            NDV = no_data_value
        in_ds = None
    arg = translate_file.replace("/", "\\"), ' -a_nodata ' + str(NDV) + ' -ot ' + \
            out_data_type + ' -mo META-TAG=VALUE' + ' -strict -q -overwrite' + ' -of ' \
            + out_driver + " -co COMPRESS=" + comp_type + " " + \
            in_ras.replace("/", "\\") + " " + out_ras.replace("/", "\\")
    subprocess.call([arg])
    return 0

def mask_ras_with_ras(in_ras, in_mask_ras, out_ras, in_band_no=1, snap_ras='inRas', \
                out_driver='input', in_ras_NDV='input', proj_chk_override=False, prec=3):
    """
    Purpose: To mask a raster given another raster as a mask. \
                    If the cell size of the inRas and inMask are unequal the \
                    function exits.
    Description of arguments:
        in_ras (string): Full path to the input raster.
        in_mask_ras (string): Full path to the raster that is used as a mask.
        out_ras (string): Full path of the output masked raster.
        snap_ras (string): The x-minimum and y-maximum coordinates of the \
                        output raster. Can be of the 'inRas' or 'maskRas'.
        out_driver (string): The short format name of the output GDAL raster \
                        driver. Defaults to that of the inRas.
        in_ras_NDV (int) : noData value of the input raster. The same is \
                        used for the output raster. Defaults to that of inRas.
    """

    in_ds = gdal.Open(in_ras, 0)
    mask_ds = gdal.Open(in_mask_ras, 0)
    if proj_chk_override is True:
        if same_coord_sys_ras_check([in_ras, in_mask_ras]) is 1:
            return
    if (in_ds == None) or (mask_ds == None):
        print('\n\a Could not open input or mask raster.')
        return
    else:
        if in_ras_NDV == 'input':
            try:
                in_ras_NDV = in_ds.GetRasterBand(in_band_no).GetNoDataValue()
            except:
                print('\n\a Could not get the noData value from the input raster (%s). '\
                    'Specify manually!' % in_ras)
                return
        else:
            in_ras_NDV = in_ras_NDV

    in_band = in_ds.GetRasterBand(in_band_no)
    in_geotransform = in_ds.GetGeoTransform()
    in_x_min = in_geotransform[0]
    in_y_max = in_geotransform[3]
    in_pix_width = in_geotransform[1]
    in_pix_height = abs(in_geotransform[5])
    in_proj = in_ds.GetProjection()
    in_coord_sys = osr.GetUserInputAsWKT(in_proj)
    in_new_data_type = in_band.DataType

    mask_band = mask_ds.GetRasterBand(1)
    mask_cols = mask_ds.RasterXSize
    mask_rows = mask_ds.RasterYSize
    mask_arr = mask_band.ReadAsArray(0,0, mask_cols, mask_rows)
    mask_NDV = mask_band.GetNoDataValue()
    mask_geotransform = mask_ds.GetGeoTransform()
    mask_x_min = mask_geotransform[0]
    mask_y_max = mask_geotransform[3]
    mask_pix_width = mask_geotransform[1]
    mask_pix_height = abs(mask_geotransform[5])
    mask_x_max = mask_x_min + (mask_pix_width * mask_cols)
    mask_y_min = mask_y_max - (mask_rows * mask_pix_height)

    if (round(mask_pix_width, prec) != round(in_pix_width, prec)) or \
                (round(mask_pix_height, prec) != round(in_pix_height, prec)):
        print('\n\a The pixel width and height of the input and masking '\
                    'rasters are unequal. Function exiting...')
        return

    x_offset = int(round(abs(mask_x_min - in_x_min)/in_pix_width))
    y_offset = int(round(abs(mask_y_max - in_y_max)/in_pix_height))
    out_cols = int(round((mask_x_max - mask_x_min) / in_pix_width))
    out_rows = int(round((mask_y_max - mask_y_min) / in_pix_height))

    if snap_ras == 'inMask':
        out_x_min = mask_x_min
        out_y_max = mask_y_max
    elif snap_ras == 'inRas':
        out_x_min = in_x_min + (x_offset * in_pix_width)
        out_y_max = in_y_max - (y_offset * in_pix_height)
    else:
        print(r"\n\a Snap raster value is not 'inMask' or 'inRas'. \
                Using Geotransform of inRas for the output. \n")
        out_x_min = in_x_min + (x_offset * in_pix_width)
        out_y_max = in_y_max - (y_offset * in_pix_height)
    if out_driver == 'input':
        out_driver = in_ds.GetDriver().ShortName
    out_arr = in_band.ReadAsArray(x_offset, y_offset, out_cols, out_rows)
    new_ras_GDAL(out_ras, out_x_min, out_y_max, out_cols, out_rows, in_pix_width, \
                in_pix_height, in_coord_sys, band_count=1, \
                out_driver=out_driver, out_data_type=in_new_data_type)
    out_ds = gdal.Open(out_ras, 1)
    out_band = out_ds.GetRasterBand(1)
    #print outArr.shape[0], outArr.shape[1], mask_arr.shape[0], mask_arr.shape[1]
    out_arr[mask_arr == mask_NDV] = in_ras_NDV
    out_band.WriteArray(out_arr, 0, 0)
    out_band.SetNoDataValue(in_ras_NDV)
    out_band.FlushCache()
    out_ds = None
    in_ds = None
    mask_ds = None
    return 0

def chng_NoDataValue(inRas, newNDV, oldNDV='input', band=1):
    """
    Purpose: To change the noData value of a given raster.
    Description of the arguments:
        inRas (string): Full path to the input raster.
        newNDV (int): The new noData value of the input raster.
        oldNDV (int): The old noData value of the input raster.
        band (int): The band for which we want to change the noData value.
    """
    inDs = gdal.Open(inRas, 1)
    inBand = inDs.GetRasterBand(band)
    outRows = inDs.RasterYSize
    outCols = inDs.RasterXSize

    if (inDs == None):
        print('\n\a Could not read input raster')
        return
    else:
        if oldNDV == 'input':
            try:
                oldNDV = inBand.GetNoDataValue()
                print(r"\n\a Original noDataValue was: %s" % oldNDV)
            except:
                print('\n\a Could not get the noData value from the input raster. \
                            Specify manually! Function Exiting...')
                return
        else:
            oldNDV = oldNDV

    # setting the entire outraster to null. Doing it in chunks because the \
                    # array might be too big
    GDALDataType = inBand.GetDataType
    GDTinBytes = GDT_to_bytes(GDALDataType)
    arrSizeInBytes = outCols * outRows * GDTinBytes
    arrSizeInMB = float(arrSizeInBytes)/ (1000000)
    sizeThresh = 750 # array size in MB if an array is larger than this \
                            # then the array is read in slices
    if arrSizeInMB > sizeThresh:
        colSlice = int(arrSizeInMB/sizeThresh)
        rowSlice = int(arrSizeInMB/sizeThresh)
        for l in range(0, (outCols/colSlice)):
            if ((colSlice * l) + colSlice) > outCols:
                #print 'if1 utilized'
                lOffset = (outCols - (colSlice * l))
            else:
                lOffset = colSlice
            for m in range(0, (outRows/rowSlice)):
                if (rowSlice * m) + rowSlice > outRows:
                    #print 'if2 utilized'
                    mOffset = (outCols - (rowSlice * m))
                else:
                    mOffset = rowSlice

                if (lOffset > 0) and (mOffset > 0):
                    #print 'OutArr arguments are: ', outCols, outRows, (colSlice * l), (rowSlice * m), lOffset, mOffset
                    inArr = inBand.ReadAsArray((colSlice * l), (rowSlice * m), lOffset, mOffset)
                    inArr[inArr == oldNDV] = newNDV
                    inBand.WriteArray(inArr, (colSlice * l), (rowSlice * m))
                else:
                    print("l or m Offset < 0 while filling raster with null values")
                    print('OutArr arguments are: ', outCols, outRows, (colSlice * l), (rowSlice * m), lOffset, mOffset)
    else:
        inArr = inBand.ReadAsArray(0, 0, outCols, outRows)
        inArr[inArr == oldNDV] = newNDV
        inBand.WriteArray(inArr, 0, 0)

    inBand.SetNoDataValue(newNDV)
    inDs.FlushCache()
    inDs = None
    return

def listHDF_subSDS(inHDF):
    """
    Purpose: To get the sub datasets of a HDF as a list.
    Description of arguments:
        inHDF (string): Full path to the input HDF file.
    """
    inDs = gdal.Open(inHDF, 0)
    try:
        sdsList = inDs.GetSubDatasets()
        return sdsList
    except:
        print(r"\n\a Could not get the SubDatasets list for the input HDF file.")
        return None

def reproj_resamp_raster(in_ds, out_ds, out_proj, in_band=1, tr_xres='input', \
            tr_yres='input', no_data_value='input', out_driver='GTiff', warp_file=warp_file, res_meth='near', comp_type='LZW'):
    """
    Purpose: To reproject and resample a given raster to a new coordinate \
                system and new given cell size using GDALWARP.
    Description of the arguments:
        in_ds (string): Full path of the input raster.
        out_ds (string): Full path of the output raster.
        in_band (int): Band number of the raster to use for masking.
        tr_xres (int or float): cell size in x-direction.
        tr_yres (int or float): cell size in y-direction.
        no_data_value (int or float): NoData value of the output raster. \
                if 'input' then the value on the input raster is used.
        warp_file (string): Full path to the GDALWARP executable. Specified in the \
                begining of this script.
        res_meth (string): The resampling method to use while reprojecting. \
                    Can be 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', \
                    'average', 'mode'. Defaults to 'near' which require th least \
                    computational effort with worst accuracy for continuous data. \
                    It is good for categorical data though.
    """
    ras = gdal.Open(in_ds, 0)
    if ras == None:
        print('Could not open input raster (%s)')
        return
    geotransform = ras.GetGeoTransform()
    if (tr_xres == 'input'):
        tr_xres = geotransform[1]
    if (tr_yres == 'input'):
        tr_yres = abs(geotransform[5])
    proj_ds = ras.GetProjection()
    proj_str = osr.GetUserInputAsWKT(proj_ds)
    ras = None
    props = list(map(str, get_ras_props(in_ds, in_band_no=1)))
    if no_data_value == 'input':
        NDV = props[9]
    else:
        NDV = str(no_data_value)
#(' -te %s %s %s %s -tr %s %s -tap' %(props[0], props[2], props[1], props[3], tr_xres, tr_yres)) + \
    arg = warp_file.replace("/", "\\"), (' -srcnodata %s  -dstnodata %s -r %s' %(NDV, NDV, res_meth)) + \
                    (' -tr %s %s -tap' %(tr_xres, tr_yres)) + \
                    ' -s_srs ' + proj_str + ' -t_srs ' + out_proj.replace("/", "\\") + \
                    " -of " + out_driver + " -co COMPRESS=" + comp_type + ' -overwrite -q -wo SKIP_NOSOURCE=YES '+ \
                    in_ds.replace("/", "\\") + " " + out_ds.replace("/", "\\")
    subprocess.call([arg])
    return 0

def clipRas_withRas(inRas, outRas, maskRas, inRasNDV='input', resMeth='near', \
            warp_file=warp_file):
    """
    Purpose: To clip a given raster to another given raster's extents using GDALWARP.
    Description of arguments:
        inRas (string): Full path of the input raster.
        outRas (string): Full path of the output raster.
        maskRas (string): Full path to the raster that is used as a mask.
        warp_file (string): Full path to the GDALWARP executable. Specified in the \
                begining of this script.
        resMeth (string): The resampling method to use while reprojecting. \
                    Can be 'near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', \
                    'average', 'mode'. Defaults to 'near' which require th least \
                    computational effort with worst accuracy for continuous data. \
                    It is good for categorical data though.

    """
    #extent = [xMin, xMax, yMin, yMax, cols, rows, pixWidth, pixHeight]
    ext = get_ras_props(maskRas)
    ras = gdal.Open(inRas, 0)
    if same_coord_sys_ras_check([inRas, maskRas]) is None:
        print('\n\a the coordinate systems of input and mask raster do not \
                    match. This might lead to undesired results. Funtion exiting...')
        return None
    if ras == None:
        print('\n\a Could not open input raster (%s)' % inRas)
        return None
    else:
        if inRasNDV == 'input':
            try:
                NDV = ras.GetRasterBand(1).GetNoDataValue()
            except:
                print('\n\a Could not get the noData value from the input raster. \
                        Specify manually!')
                return None
        else:
            NDV = inRasNDV
        ras = None
    arg = warp_file.replace("/", "\\"), ' -srcnodata ' + str(NDV) + ' -dstnodata ' \
            + str(NDV) + ' -overwrite ' + '-te ' + str(ext[0]) + " " + str(ext[2]) + \
            " " + str(ext[1]) + " " + str(ext[3]) + ' -r ' + resMeth  + " " + \
            inRas.replace("/", "\\") + " " + outRas.replace("/", "\\")
    subprocess.call([arg])
    return

def resample_ras_scipy_GDAL_withRaster(inRas, outRas, outXRes, outYRes, order=3, \
    inBand=1, noDataValue='input', maxLim='input', minLim='input'):

    """
    Purpose: To resample a given band of a raster to a given output resolution \
                using a given method. This function is not tested throughly and \
                is only suited for images with no noData values.
    Description of arguments:
        inRas (string): Full path to the input raster.
        outRas (string): Full path of the output raster.
        outXRes (int): Required output cell size in x-direction.
        outYRes (int): Required output cell size in y-direction.
        order (int): Corresponds to the resampling method (order of spline \
                interpolation). 0 for 'nearest' method and \
                1 for 'bi-linear', 2 for 'bi-quadratic', 3 for 'bi-cubic', \
                4 for 'bi-quartic', 5 for 'bi-quintic'.
        inBand (int): The band number in the raster that we want to resample.
        noDataValue (int or float): noData value of the output raster. Defaults \
                to that of the input.
        maxLim (int or float): The maximum number in the output raster \
                that we think is possible. Interpolations between noData values \
                and a other numbers can give numbers which are not possible.
        minLim (int or float): The lower limit of numbers in the output raster. \
                Same reason as maxLim.
    """
    ext = get_ras_props(inRas) # [xMin, xMax, yMin, yMax, cols, rows, pixWidth, pixHeight]
    inDs = gdal.Open(inRas, 1)
    if inDs == None:
        print('\n\a Could not open input raster (%s)' % inRas)
        return None
    else:
        if noDataValue == 'input':
            try:
                NDV = inDs.GetRasterBand(1).GetNoDataValue()
            except:
                print('\n\a Could not get the noData value from the input raster. Specify manually!')
                return None
        else:
            NDV = noDataValue
    inBand = inDs.GetRasterBand(inBand)
    zoomX = float(ext[6])/outXRes
    zoomY = float(ext[7])/outYRes
    inArr = inBand.ReadAsArray(0,0, ext[4], ext[5])

    proj_in = inDs.GetProjection()
    projStr_in = osr.GetUserInputAsWKT(proj_in)
    driver = inDs.GetDriver().ShortName
    in_newDataType = inBand.DataType

    new_ras_GDAL(outRas, ext[0], ext[3], int(round((ext[4] * zoomX))), int(round(ext[5] * zoomY)), \
                    outXRes, outYRes, projStr_in, bandCount=1, driver=driver, dataType=in_newDataType)
    outDs = gdal.Open(outRas, 1)
    outBand = outDs.GetRasterBand(1)
    outArr = spni.zoom(inArr, zoom=(zoomX, zoomY), order=order) \
        # check http://scikit-image.org/docs/dev/api/skimage.transform.html#warp
        # check http://scikit-image.org/docs/dev/api/skimage.transform.html#resize
        # check http://scikit-image.org/docs/dev/api/skimage.transform.html#rescale
        # check http://effbot.org/imagingbook/image.htm#tag-Image.Image.resize

    if maxLim == 'input':
        outArr[outArr > np.max(np.ma.masked_equal(inArr, NDV))] = NDV
    else:
        outArr[outArr > maxLim] = NDV

    if minLim == 'input':
        outArr[outArr < np.min(np.ma.masked_equal(inArr, NDV))] = NDV
    else:
        outArr[outArr < minLim] = NDV

    outBand.WriteArray(outArr, 0, 0)
    outBand.SetNoDataValue(NDV)
    outBand.FlushCache()
    outDs = None
    inDs = None
    return

def resample_ASCII_scipy(inASCIIFile, outASCIIFile, outRes, order=3, prec=3, \
                    maxLim='input', minLim='input'):
    """
    Purpose: To resample and ASCII text file using scipy. \
                This function works for array with no noData values.
    Description of arguments:
        inASCIIFile (string): Full path to the input ASCIIFile that \
                        we want to resample.
        outASCIIFile (string): Full path of the resampled output ASCIIFile. \
                        It's deleted if it exists already.
        outRes (int or float): The output cell size of the outASCIIFile.
        order (int): Corresponds to the resampling method (order of spline \
                interpolation). 0 for 'nearest' method and \
                1 for 'bi-linear', 2 for 'bi-quadratic', 3 for 'bi-cubic', \
                4 for 'bi-quartic', 5 for 'bi-quintic'.
        prec (int):  Decimal point precision of the values in outASCIIFile.
        maxLim (int or float): The maximum number in the output raster \
                that we think is possible. Interpolations between noData values \
                and a other numbers can give numbers which are not possible.
        minLim (int or float): The lower limit of numbers in the output raster. \
                Same reason as maxLim.
    """

    if not os.path.exists(inASCIIFile):
        print('\n\a Could not read input ASCII file (%s)' % inASCIIFile)
        return None
    else:
        xLeftCorner = float(linecache.getline(inASCIIFile, 3).split()[1])
        cellSize = float(linecache.getline(inASCIIFile, 5).split()[1])
        noDataValue = float(linecache.getline(inASCIIFile, 6).split()[1])
        yLowCorner = float(linecache.getline(inASCIIFile, 4).split()[1])

        zoom = float(cellSize)/outRes
        inArr = np.loadtxt(inASCIIFile, dtype=np.float, skiprows=6)
        #inArr[inArr==noDataValue] = np.nan
        outArr = spni.zoom(inArr, zoom, order=order)

        if maxLim == 'input':
            outArr[outArr > np.max(np.ma.masked_equal(inArr, noDataValue))] = noDataValue
        else:
            outArr[outArr > maxLim] = noDataValue

        if minLim == 'input':
            outArr[outArr < np.min(np.ma.masked_equal(inArr, noDataValue))] = noDataValue
        else:
            outArr[outArr < minLim] = noDataValue
        if os.path.exists(outASCIIFile):
            os.remove(outASCIIFile)
        cursor = open(outASCIIFile, 'w')
        cursor.write('ncols\t' + str(outArr.shape[1]) + '\n')
        cursor.write('nrows\t' + str(outArr.shape[0]) + '\n')
        cursor.write('xllcorner\t' + str(xLeftCorner) + '\n')
        cursor.write('yllcorner\t' + str(yLowCorner) + '\n')
        cursor.write('cellsize\t' + str(outRes) + '\n')
        cursor.write('NODATA_value\t' + str(noDataValue)+ '\n')
        np.savetxt(cursor, outArr, fmt='%1.' + str(prec) + 'f')
        cursor.close()
        return

def maskASCII_using_Ras(inASCIIFile, inMaskRas, outASCIIFile, prec=3):
    """
    Purpose: To mask an ASCIIFile using a raster. Every parameter (extents \
                and cell size) of the inMaskRas and inASCIIFile should be \
                the same otherwise the script will give an error.
    Description of arguments:
        inASCIIFile (string): Full path to the input ASCII file.
        inMaskRas (string): Full path to the masking raster.
        outASCIIFile (string): Full path of the output ASCII file. Deleted if \
                it exists already.
        prec (int): Decimal point precision of the values in the outASCIIFile.
    """
    print('\n\a ')
    maskDs = gdal.Open(inMaskRas, 0)
    if maskDs != None:
        mask_band = maskDs.GetRasterBand(1)
        mask_cols = maskDs.RasterXSize
        mask_rows = maskDs.RasterYSize
        mask_arr = mask_band.ReadAsArray(0,0, mask_cols, mask_rows)
        mask_NDV = mask_band.GetNoDataValue()
        mask_geotransform = maskDs.GetGeoTransform()
        mask_xMin = mask_geotransform[0]
        mask_yMax = mask_geotransform[3]
        mask_pixWidth = mask_geotransform[1]
        mask_pixHeight = abs(mask_geotransform[5])
        mask_yMin = mask_yMax - (mask_rows * mask_pixHeight)
    else:
        print(r"\n\a Cannot read the input mask raster (%s). \
                    Do something about it." % inMaskRas)
        return None

    if not os.path.exists(inASCIIFile):
        print('\n\a Could not read input ASCII file (%s).' % inASCIIFile)
        return None
    else:
        xLeftCorner = float(linecache.getline(inASCIIFile, 3).split()[1])
        cellSize = float(linecache.getline(inASCIIFile, 5).split()[1])
        noDataValue = float(linecache.getline(inASCIIFile, 6).split()[1])
        yLowCorner = float(linecache.getline(inASCIIFile, 4).split()[1])

    if (cellSize != mask_pixWidth) or (cellSize != mask_pixHeight) or \
                (xLeftCorner != mask_xMin) or (yLowCorner != mask_yMin) :
        print('\n\a Cellsize or x-Min and y-Max of mask and inASCII file \
                        are not the same. Exiting function...')
        return None

    inArr = np.loadtxt(inASCIIFile, skiprows=6)
    inArr[mask_arr == mask_NDV] = noDataValue

    if os.path.exists(outASCIIFile):
        os.remove(outASCIIFile)
    cursor = open(outASCIIFile, 'w')
    cursor.write('ncols\t' + str(inArr.shape[1]) + '\n')
    cursor.write('nrows\t' + str(inArr.shape[0]) + '\n')
    cursor.write('xllcorner\t' + str(xLeftCorner) + '\n')
    cursor.write('yllcorner\t' + str(yLowCorner) + '\n')
    cursor.write('cellsize\t' + str(cellSize) + '\n')
    cursor.write('NODATA_value\t' + str(noDataValue)+ '\n')
    np.savetxt(cursor, inArr, fmt='%1.' + str(prec) + 'f')
    cursor.close()
    return

def str_to_GDT(inDT):
    """
    Purpose: To return a GDAL data type corresponding to a data type's name.
    Description of arguments:
        inDT (string): The input data type name in string. Can be 'float32', \
                float64, 'int32', 'uint32', 'int16', 'uint16', 'int'.
    """
    inDT = inDT.lower()
    if inDT == 'float32':
        outGDT = gdal.GDT_Float32
    elif inDT == 'float64':
        outGDT = gdal.GDT_Float64
    elif inDT == 'int32':
        outGDT = gdal.GDT_Int32
    elif inDT == 'uint32':
        outGDT = gdal.GDT_UInt32
    elif inDT == 'int16':
        outGDT = gdal.GDT_Int16
    elif inDT == 'uint16':
        outGDT = gdal.GDT_UInt16
    elif inDT == 'int':
        outGDT = gdal.GDT_Byte
    else:
        print('\n\a The specified data type cannot be returned as GDAL data type. \n\
                The input can be int, uint16, int16, uint32, int32, float64, float32.')
        return None
    return outGDT

def NPDTtoGDT (inNDT):
    """
    Purpose: To return a GDAL data type corresponding to a numpy data type.
    Description of arguments:
        inNDT (numpy data type): The input numpy data type. Can be 'np.float32', \
                np.float64, 'np.int32', 'np.uint32', 'np.int16', 'np.uint16', 'np.int'.
    """
    if inNDT == np.float32:
        outGDT = gdal.GDT_Float32
    elif inNDT == np.float64:
        outGDT = gdal.GDT_Float64
    elif inNDT == np.int32:
        outGDT = gdal.GDT_Int32
    elif inNDT == np.uint32:
        outGDT = gdal.GDT_UInt32
    elif inNDT == np.int16:
        outGDT = gdal.GDT_Int16
    elif inNDT == np.uint16:
        outGDT = gdal.GDT_UInt16
    elif inNDT == np.int8:
        outGDT = gdal.GDT_Byte
    else:
        print('\n\a The specified numpy data type cannot be returned as GDAL data type. \n\
                The input can be np.int8, np.uint16, np.int16, np.uint32, np.int32, \
                np.float64, np.float32.')
        return None
    return outGDT

def strToNPDT (inDT):
    """
    Purpose: To return a numpy data type corresponding to a data type's name.
    Description of arguments:
        inDT (string): The input data type name in string. Can be 'float32', \
                float64, 'int32', 'uint32', 'int16', 'uint16', 'int'.
    """
    inDT = inDT.lower
    if inDT == 'float32':
        outNDT = np.float32
    elif inDT == 'float64':
        outNDT = np.float64
    elif inDT == 'int32':
        outNDT = np.int32
    elif inDT == 'uint32':
        outNDT = np.uint32
    elif inDT == 'int16':
        outNDT = np.int16
    elif inDT == 'uint16':
        outNDT = np.uint16
    elif inDT == 'int':
        outNDT = np.int8
    else:
        print('\n\a The specified data type cannot be returned as GDAL datatype. \n\
                The input can be int, uint16, int16, uint32, int32, float64, float32.')
        return None
    return outNDT

def GDT_to_bytes(inGDT):
    """
    Purpose: To return a GDAL data type's size in bytes.
    Description of arguments:
        inGDT (GDAl data type): The input GDAL data type. Can be gdal.GDT_Byte, \
                gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, \
                gdal.GDT_Float64, gdal.GDT_Float32.'
    """
    if inGDT == gdal.GDT_Float32:
        bytes = 4
    elif inGDT == gdal.GDT_Float64:
        bytes = 8
    elif inGDT == gdal.GDT_Int32:
        bytes = 4
    elif inGDT == gdal.GDT_UInt32:
        bytes = 4
    elif inGDT == gdal.GDT_Int16:
        bytes = 2
    elif inGDT == gdal.GDT_UInt16:
        bytes =   2
    elif inGDT == gdal.GDT_Byte:
        bytes = 1
    else:
        print('\n\a The size of the specified data type cannot be returned in bytes. \n\
                The input can be gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, \
                gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float64, gdal.GDT_Float32.')
        return None
    return bytes

def GDT_to_NPDT(inGDT):
    """
    Purpose: To return a numpy data type corresponding to a GDAL data type.
    Description of arguments:
        inGDT (numpy data type): The input GDAL data type. Can be gdal.GDT_Byte, \
                    gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, \
                    gdal.GDT_Int32, gdal.GDT_Float64, gdal.GDT_Float32.
    """
    if inGDT == gdal.GDT_Float32:
        outNDT = np.float32
    elif inGDT == gdal.GDT_Float64:
        outNDT = np.float64
    elif inGDT == gdal.GDT_Int32:
        outNDT = np.int32
    elif inGDT == gdal.GDT_UInt32:
        outNDT = np.uint32
    elif inGDT == gdal.GDT_Int16:
        outNDT = np.int16
    elif inGDT == gdal.GDT_UInt16:
        outNDT = np.uint16
    elif inGDT == gdal.GDT_Byte:
        outNDT = np.int8
    else:
        print('\n\a The specified GDAL data type cannot be returned as numpy datatype. \n\
                The input can be gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, \
                gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float64, gdal.GDT_Float32.')
        return None
    return outNDT

def new_ras_from_given_ras_GDAL(in_ras, out_ras, in_band_no=1, \
                out_driver='input', data_type='input', comp_type='LZW'):
    """
    Purpose: To create a new raster given the required parameters
    Description of the arguments:
        outRas (string): Full path to the output raster. The driver to create \
                        the outRas is specified with the 'driver' argument.
        driver (string, short format name): Short name of the file system \
                        according to GDAL. Defaults to GTiff.
        dataType (GDAL datatype): Corresponds to GDAL data type. Sometimes, while \
                        running the function an error may occur saying that \
                        there is no module named GDAL, in that case use the \
                        'str_to_GDT' function.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds is not None:
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize

        geotransform = in_ds.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        proj = in_ds.GetProjection()

        band_count = in_ds.RasterCount

        if out_driver == 'input':
            out_driver = in_ds.GetDriver()
        else:
            out_driver = gdal.GetDriverByName(out_driver)

        if data_type == 'input':
                data_type = in_ds.GetRasterBand(in_band_no).DataType

        out_ds = out_driver.Create(out_ras, cols, rows, band_count, \
                    data_type, options=['COMPRESS='+comp_type])
        out_ds.SetGeoTransform([x_min, pix_width, 0, y_max, 0, -pix_height])
        proj = osr.GetUserInputAsWKT(proj)
        out_ds.SetProjection(proj)
        out_ds = in_ds = None
        return
    else:
        print('\n\a Could not read the input raster (%s). Check path and file!' % in_ras)
        return None

def hargreaves_pet(d_o_y, lat, t_min, t_max, t_avg):
    """
    Purpose: To get the potential evapotranspiration at a given latitude \
                for a given date and temperature.
    Description of the arguments:
        d_o_y (int): day of the year.
        lat (degrees): latitude of the point.
        t_min (celsius): minimum temperature on that day
        t_max (celsius): maximum temperature on that day
        t_avg (celsius): average temperature on that day
    """

    tot_days = 365

    lat = radians(lat)
    ndec = 0.409 * sin(((2 * pi * d_o_y) / tot_days) - 1.39)
    nws = acos(-tan(lat) * tan(ndec))
    dfr = 1 + (0.033 * cos((2 * pi * d_o_y) / tot_days))

    fac_1 = 15.342618001389575 # ((1440 * 0.082 * 0.4082)/pi)

    ra = fac_1 * dfr * ((nws * sin(lat) * sin(ndec)) + \
                (cos(lat) * cos(ndec) * sin(nws)))

    fac_2 = 0.002295 # (0.0135 * 0.17)

    pet = fac_2 * ra * sqrt(t_max-t_min) * (t_avg + 17.8)

    if pet > 0:
        return pet
    else:
        return 0


def change_pt_crs(x, y, in_epsg, out_epsg):
    """
    Purpose:
        To return the coordinates of given points in a different coordinate system.

    Description of arguments:
        x (int or float, single or list): The horizontal position of the input point
        y (int or float, single or list): The vertical position of the input point
        Note: In case of x and y in list form, the output is also in a list form.
        in_epsg (string or int): The EPSG code of the input coordinate system
        out_epsg (string or int): The EPSG code of the output coordinate system
    """
    in_crs = pyproj.Proj("+init=EPSG:" + str(in_epsg))
    out_crs = pyproj.Proj("+init=EPSG:" + str(out_epsg))
    return pyproj.transform(in_crs, out_crs, float(x), float(y))

def shp_to_ras(in_shp, # full path to the input vector file
               out_ras, # full path of the output raster file
               x_res, # out horizontal cell size
               y_res, # output vertical cell size
               out_ext=None, # output extents (in list form)
                                   # [xmin, ymin, xmax, ymax]
               burn_value=1, # a value for the cells that
                                   # intersect the objects
                                   # in the shapefile
               burn_atbt=None, # attribute filed name in
                                   # the vector file to use
                                   # as the burn value,
                                   # if None then bun_value is used
               out_NDV=-32678, # output no data value
               out_DT='Float64', # output raster data type
               out_driver='GTiff', # ouput raster driver (shortname)
               at=True, # enable all touched mode
               comp_type='LZW' # type of compression for the output raster
               ):

    arg = ''

    arg += ' -tr %f %f' %(x_res, y_res)

    if out_ext is not None:
        if type(out_ext) is list:
            arg += ' -te %f %f %f %f' % (out_ext[0],
                                         out_ext[1],
                                         out_ext[2],
                                         out_ext[3])
        else:
            print('Output extents of the raster are ' + \
                        'not provided as a list. Function cannot proceed.')
            return

    if burn_atbt is None:
        arg += ' -burn %f' % burn_value
    else:
        arg += ' -a %s' % burn_atbt

    arg += ' -a_nodata {0} -init {0}'.format(out_NDV)

    arg += ' -ot %s' % out_DT

    arg += ' -of %s' % out_driver

    if at is True:
        arg += ' -at'

    arg += ' -co COMPRESS=%s' % comp_type

    arg += ' -q'

    arg += ' %s %s' % (in_shp.replace("/", "\\"),
                       out_ras.replace("/", "\\"))

    arg = rasterize_file.replace("/", "\\"), arg

    try:
        subprocess.call([arg])
        return 0
    except:
        print('Something went wrong while rasterizing the vector file (%s). ' +\
                    ' Rasterizing failed.' % in_shp)
        return

def cnvt_to_pt(x, y):
    """Convert x y coordinates to a point string
    in POINT(x y) format
    """
    pt = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (x, y))
    return pt


def chk_cntmt(pt, bbx):
    """Containment check of points in a given polygon
    """
    cntmt = bbx.Contains(pt)
    return cntmt


#@jit()
def zero_ftn(x):
    return np.where(x == 0)[0]


#@jit()
def get_sum(x):
    return np.sum(x)


#@jit()
def round_it(x):
    return round(x, 2)


#@jit
def get_dist(x1, y1, x2, y2):
    """Get distance between points
    """
    dist = (((x1 - x2)**2 + (y1 - y2)**2))**0.5
    return dist

#@jit
def get_idw(dists, vals, idw_exp):
    """Get IDW value at a point given distances of
    other points from it with values and the exponent
    of the IDW
    """
    zero_idx = zero_ftn(dists)
    if zero_idx.shape[0] > 1:
        idw_val = vals[zero_idx]
        return idw_val
    else:
        idw_wts = 1 / np.power(dists, idw_exp)
        the_sum = get_sum(idw_wts * vals)
        the_wts = get_sum(idw_wts)
        idw_val = the_sum / the_wts
        return idw_val


def ppt_idw_mp(in_ppt_df, in_coords_df, out_coords_df, idw_exp, que):
    ''' Get IDW values at other stations given input stations
    '''

    def out_ppt_idw(out_stn, idw_exp):
        x = out_coords_df.loc[out_stn]['X']
        y = out_coords_df.loc[out_stn]['Y']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_ppt_vals, idw_exp)

        if idw_val < 0.1:
            idw_val = 0
        else:
            idw_val = round(idw_val, 2)

        out_ppt_df.loc[date][out_stn] = idw_val
        return None


    out_ppt_df = pd.DataFrame(index=in_ppt_df.index, columns=out_coords_df.index)

    for date in in_ppt_df.index:
        in_ppt_ser = in_ppt_df.loc[date]
        avail_ppt_stn_list = []

        for idx in in_ppt_ser.index:
            if not pd.np.isnan(in_ppt_ser.loc[idx]):
                avail_ppt_stn_list.append(idx)

        xs = in_coords_df.loc[avail_ppt_stn_list]['X']
        ys = in_coords_df.loc[avail_ppt_stn_list]['Y']
        avail_ppt_vals = in_ppt_ser.loc[avail_ppt_stn_list].values

        pd.np.vectorize(out_ppt_idw)(out_coords_df.index, idw_exp)

    que.put(out_ppt_df)
    return None


def min_temp_mp(arg):
    min_temp_df, hist_coords_df, in_virt_stn_coords_df, min_temp_virt_df, temp_idw_exp, elev_idw_exp, min_temp_q = arg

    def min_virt_temp_idw(virt_stn):
        x = in_virt_stn_coords_df.loc[virt_stn]['X']
        y = in_virt_stn_coords_df.loc[virt_stn]['Y']
        z = in_virt_stn_coords_df.loc[virt_stn]['Z']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_temp_vals, temp_idw_exp)
        elev_idw_val = get_idw(dists, zs, elev_idw_exp)
        elev_diff = z - elev_idw_val
        corr =  elev_diff * (-0.0064)
        idw_val += corr
        min_temp_virt_df.loc[date][virt_stn] = idw_val
        return None

    for date in min_temp_df.index:
        min_temp_ser = min_temp_df.loc[date]
        avail_temp_stn_list = []

        for idx in min_temp_ser.index:
            if not pd.np.isnan(min_temp_ser.loc[idx]):
                avail_temp_stn_list.append(idx)

        xs = hist_coords_df.loc[avail_temp_stn_list]['X']
        ys = hist_coords_df.loc[avail_temp_stn_list]['Y']
        zs = hist_coords_df.loc[avail_temp_stn_list]['Z']
        avail_temp_vals = min_temp_ser.loc[avail_temp_stn_list].values

        pd.np.vectorize(min_virt_temp_idw)(in_virt_stn_coords_df.index)

    min_temp_q.put([min_temp_df, min_temp_virt_df])
    return


def max_temp_mp(arg):
    max_temp_df, hist_coords_df, in_virt_stn_coords_df, max_temp_virt_df, temp_idw_exp, elev_idw_exp, max_temp_q = arg

    def max_virt_temp_idw(virt_stn):
        x = in_virt_stn_coords_df.loc[virt_stn]['X']
        y = in_virt_stn_coords_df.loc[virt_stn]['Y']
        z = in_virt_stn_coords_df.loc[virt_stn]['Z']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_temp_vals, temp_idw_exp)
        elev_idw_val = get_idw(dists, zs, elev_idw_exp)
        elev_diff = z - elev_idw_val
        corr =  elev_diff * (-0.0064)
        idw_val += corr
        max_temp_virt_df.loc[date][virt_stn] = idw_val
        return None

    for date in max_temp_df.index:
        max_temp_ser = max_temp_df.loc[date]
        avail_temp_stn_list = []

        for idx in max_temp_ser.index:
            if not pd.np.isnan(max_temp_ser.loc[idx]):
                avail_temp_stn_list.append(idx)

        xs = hist_coords_df.loc[avail_temp_stn_list]['X']
        ys = hist_coords_df.loc[avail_temp_stn_list]['Y']
        zs = hist_coords_df.loc[avail_temp_stn_list]['Z']
        avail_temp_vals = max_temp_ser.loc[avail_temp_stn_list].values

        pd.np.vectorize(max_virt_temp_idw)(in_virt_stn_coords_df.index)

    max_temp_q.put([max_temp_df, max_temp_virt_df])
    return


def avg_temp_mp(arg):
    avg_temp_df, hist_coords_df, in_virt_stn_coords_df, avg_temp_virt_df, temp_idw_exp, elev_idw_exp, avg_temp_q = arg

    def avg_virt_temp_idw(virt_stn):
        x = in_virt_stn_coords_df.loc[virt_stn]['X']
        y = in_virt_stn_coords_df.loc[virt_stn]['Y']
        z = in_virt_stn_coords_df.loc[virt_stn]['Z']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_temp_vals, temp_idw_exp)
        elev_idw_val = get_idw(dists, zs, elev_idw_exp)
        elev_diff = z - elev_idw_val
        corr =  elev_diff * (-0.0064)
        idw_val += corr
        avg_temp_virt_df.loc[date][virt_stn] = idw_val
        return None

    for date in avg_temp_df.index:
        avg_temp_ser = avg_temp_df.loc[date]
        avail_temp_stn_list = []

        for idx in avg_temp_ser.index:
            if not pd.np.isnan(avg_temp_ser.loc[idx]):
                avail_temp_stn_list.append(idx)

        xs = hist_coords_df.loc[avail_temp_stn_list]['X']
        ys = hist_coords_df.loc[avail_temp_stn_list]['Y']
        zs = hist_coords_df.loc[avail_temp_stn_list]['Z']
        avail_temp_vals = avg_temp_ser.loc[avail_temp_stn_list].values

        pd.np.vectorize(avg_virt_temp_idw)(in_virt_stn_coords_df.index)

    avg_temp_q.put([avg_temp_df, avg_temp_virt_df])
    return


def pet_harg_mp(min_temp_df, max_temp_df, avg_temp_df, lat_arr, pet_que):
    pet_df = pd.DataFrame(index=min_temp_df.index, columns=min_temp_df.columns)

    for i in range(min_temp_df.shape[0]):
        doy = min_temp_df.index[i].dayofyear
        t_min_arr = min_temp_df.iloc[i].values
        t_max_arr = max_temp_df.iloc[i].values
        t_avg_arr = avg_temp_df.iloc[i].values
        pet_df.iloc[i] = pd.np.vectorize(hargreaves_pet)(doy, lat_arr, t_min_arr, t_max_arr, t_avg_arr)

    pet_que.put(pet_df)
    return None


def ppt_idw(in_ppt_df, in_coords_df, out_ppt_df, out_coords_df, idw_exp):
    ''' Get IDW values at other stations given input stations
    '''

    def out_ppt_idw(out_stn, idw_exp):
        x = out_coords_df.loc[out_stn]['X']
        y = out_coords_df.loc[out_stn]['Y']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_ppt_vals, idw_exp)

        if idw_val < 0.1:
            idw_val = 0
        else:
            idw_val = round(idw_val, 2)

        out_ppt_df.loc[date][out_stn] = idw_val
        return None


    for date in in_ppt_df.index:
        in_ppt_ser = in_ppt_df.loc[date]
        avail_ppt_stn_list = []

        for idx in in_ppt_ser.index:
            if not pd.np.isnan(in_ppt_ser.loc[idx]):
                avail_ppt_stn_list.append(idx)

        xs = in_coords_df.loc[avail_ppt_stn_list]['X']
        ys = in_coords_df.loc[avail_ppt_stn_list]['Y']
        avail_ppt_vals = in_ppt_ser.loc[avail_ppt_stn_list].values

        pd.np.vectorize(out_ppt_idw)(out_coords_df.index, idw_exp)

    return out_ppt_df

def ppt_idw_valid(in_ppt_df, in_coords_df, out_coords_df, in_valid_ctrl_df, idw_exp):
    ''' Get IDW values at other stations given input stations with ctrl and valid data frame
    '''

    def out_ppt_idw(out_stn, idw_exp):
        x = out_coords_df.loc[out_stn]['X']
        y = out_coords_df.loc[out_stn]['Y']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_ppt_vals, idw_exp)

        if idw_val < 0.1:
            idw_val = 0
        else:
            idw_val = round(idw_val, 2)

        out_ppt_df.loc[date][out_stn] = idw_val
        return None

    out_ppt_df = pd.DataFrame(index=in_ppt_df.index, columns=in_ppt_df.columns)

    for date in in_ppt_df.index:
        in_ppt_ser = in_ppt_df.loc[date]
        avail_ppt_stn_list = []

        dum_stns = in_valid_ctrl_df.loc[date].values
        ctrl_stns = in_valid_ctrl_df.loc[date].index[np.where(dum_stns=='ctrl_stn')[0]]

        dum_valids = in_valid_ctrl_df.loc[date].values
        valid_stns = in_valid_ctrl_df.loc[date].index[np.where(dum_valids=='valid_stn')[0]]

        if len(valid_stns) == 0:
            continue

        for idx in in_ppt_ser.index:
            nan_cond = (not np.isnan(in_ppt_ser.loc[idx]))
            ctrl_cond = (idx in ctrl_stns)
            if nan_cond and ctrl_cond:
                avail_ppt_stn_list.append(idx)

        if len(avail_ppt_stn_list) == 0:
            continue

        xs = in_coords_df.loc[avail_ppt_stn_list]['X']
        ys = in_coords_df.loc[avail_ppt_stn_list]['Y']
        avail_ppt_vals = in_ppt_ser.loc[avail_ppt_stn_list].values

        pd.np.vectorize(out_ppt_idw)(valid_stns, idw_exp)

    return out_ppt_df.dropna(axis=1, how='all')

def ppt_idw_valid_mp(in_ppt_df, in_coords_df, out_coords_df, in_valid_ctrl_df, idw_exp, valid_que):
    ''' Get IDW values at other stations given input stations with ctrl and valid data frame
    '''

    def out_ppt_idw(out_stn, idw_exp):
        x = out_coords_df.loc[out_stn]['X']
        y = out_coords_df.loc[out_stn]['Y']
        dists = pd.np.vectorize(get_dist)(x, y, xs, ys)
        idw_val = get_idw(dists, avail_ppt_vals, idw_exp)

        if idw_val < 0.1:
            idw_val = 0
        else:
            idw_val = round(idw_val, 2)

        out_ppt_df.loc[date][out_stn] = idw_val
        return None

    out_ppt_df = pd.DataFrame(index=in_ppt_df.index, columns=in_ppt_df.columns)

    for date in in_ppt_df.index:
        in_ppt_ser = in_ppt_df.loc[date]
        avail_ppt_stn_list = []

        dum_stns = in_valid_ctrl_df.loc[date].values
        ctrl_stns = in_valid_ctrl_df.loc[date].index[np.where(dum_stns=='ctrl_stn')[0]]

        dum_valids = in_valid_ctrl_df.loc[date].values
        valid_stns = in_valid_ctrl_df.loc[date].index[np.where(dum_valids=='valid_stn')[0]]

        if len(valid_stns) == 0:
            continue

        for idx in in_ppt_ser.index:
            nan_cond = (not np.isnan(in_ppt_ser.loc[idx]))
            ctrl_cond = (idx in ctrl_stns)
            if nan_cond and ctrl_cond:
                avail_ppt_stn_list.append(idx)

        if len(avail_ppt_stn_list) == 0:
            continue

        xs = in_coords_df.loc[avail_ppt_stn_list]['X']
        ys = in_coords_df.loc[avail_ppt_stn_list]['Y']
        avail_ppt_vals = in_ppt_ser.loc[avail_ppt_stn_list].values

        pd.np.vectorize(out_ppt_idw)(valid_stns, idw_exp)


    valid_que.put(out_ppt_df)
    return None

def plot_act_interp_ppt_mp(in_act_ppt_df, in_interp_ppt_df, out_loc):
    def plot_compare_figs(width=0.35):
        plt.figure(figsize=(20, 10))

        act_ppt_vals = act_ser.values
        interp_ppt_vals = interp_ser.values
        index_vals = np.arange(0, act_ppt_vals.shape[0])

        plt.bar(index_vals, act_ppt_vals, width=width, color='b', alpha=0.7, label='Actual Precipitation')
        plt.bar(index_vals+width, interp_ppt_vals, width=width, color='r', alpha=0.7, label='Interpolated Precipitation')

        plt.xticks(index_vals + width, union_stns)
        plt.xlabel('Station')
        plt.ylabel('Precipitation (mm)')
        plt.grid()

        plt.legend(loc=0)

        plt.savefig(out_plot_loc % 'validation_comparison', dpi=150, bbox='tight_layout')

        plt.clf()
        plt.close()

        return None

    out_plots_loc = out_loc + r'{:s}_{:0>4d}.{:0>2d}.{:0>2d}.png'
    for date in in_act_ppt_df.index:
        union_stns = np.union1d(in_act_ppt_df.loc[date].dropna().index, in_interp_ppt_df.loc[date].dropna().index)
        if union_stns.shape[0] == 0:
            continue

        act_ser = in_act_ppt_df.loc[date][union_stns]
        interp_ser = in_interp_ppt_df.loc[date][union_stns]

        out_plot_loc = out_plots_loc.format(r'%s', date.year, date.month, date.day)

        plot_compare_figs()

    return None


def ASCII_idw_grid(x,
                   y,
                   z,
                   x_min,
                   x_max,
                   y_min,
                   y_max,
                   cell_size,
                   idw_exp,
                   out_loc):

    """Interpoalte an IDW grid
    """

    out_rows = int(np.ceil((y_max - y_min) / cell_size))
    out_cols = int(np.ceil((x_max - x_min) / cell_size))

    in_rows = (y_max - y) / cell_size
    in_cols = (x - x_min) / cell_size

    idw_grid = np.empty((out_rows, out_cols))
    idw_rows = np.arange(0.5, out_rows, 1)
    idw_cols = np.arange(0.5, out_cols, 1)

    for i in range(idw_grid.shape[0]):
        for j in range(idw_grid.shape[1]):
            dists = np.vectorize(get_dist)(idw_cols[j], idw_rows[i], in_cols, in_rows)
            idw_grid[i, j] = get_idw(dists, z, idw_exp)

    cursor = open(out_loc, 'w')
    cursor.write('ncols\t' + str(out_cols) + '\n')
    cursor.write('nrows\t' + str(out_rows) + '\n')
    cursor.write('xllcorner\t' + str(x_min) + '\n')
    cursor.write('yllcorner\t' + str(y_min) + '\n')
    cursor.write('cellsize\t' + str(cell_size) + '\n')
    cursor.write('NODATA_value\t' + str(-9999) + '\n')
    np.savetxt(cursor, idw_grid, fmt='%1.' + str(15) + 'f')
    cursor.close()

    return None


def ASCII_idw_grid_mp(in_ppt_df,
                      in_coords_df,
                      cell_size_ser,
                      idw_exp,
                      x_min,
                      x_max,
                      y_min,
                      y_max,
                      out_ASCII_loc,
                      out_pref):

    for date in in_ppt_df.index:
        ppt_ser = in_ppt_df.loc[date].dropna().copy()
        xs = in_coords_df.loc[ppt_ser.index]['X']
        ys = in_coords_df.loc[ppt_ser.index]['Y']
        cell_size = cell_size_ser.loc[date]
        zs = ppt_ser.values
        out_name = r'%s_%0.4d.%0.2d.%0.2d.csv' % (out_pref, date.year, date.month, date.day)
        out_loc = out_ASCII_loc + out_name
        ASCII_idw_grid(xs, ys, zs, x_min, x_max, y_min, y_max, cell_size, idw_exp, out_loc)

    return None


def get_cell_size(x, y):
    x, y = pd.np.ravel(x), pd.np.ravel(y)
    cell_size_adj = 0.99
    xx1, xx2 = pd.np.meshgrid(x, x)
    yy1, yy2 = pd.np.meshgrid(y, y)
    dists = pd.np.sqrt((xx2-xx1)**2 + (yy2-yy1)**2)
    cell_size = cell_size_adj*pd.np.sqrt(pow(pd.np.unique(dists)[1], 2) / 2)
    return float(cell_size)


def get_sill_and_range(model):
    models = model.split('+')
    models = pd.np.array(models)
    Sill = 0.0
    # go through models:
    for submodel in models:
        submodel = submodel.strip()
        Sill  += float(submodel.split('(')[0].strip()[:-3].strip())
        Range = submodel.split('(')[1].split(')')[0]
    return [float(Sill), float(Range)]


def adj_vg_str(vg_str, new_sill_val=1., rang_fac=1):
    """To adjust the total sill value and range of a given variogram string

    Parameters:
        vg_str: The input variogram string with a format: Sill Type(Range)
                Can be multiple e.g. Sill1 Type1(Range1) + Sill2 Type2(Range2)

        new_sill_val: The new total sill value. All sills are adjusted based on ratios

        rang_fac: The factor with which to increase of decrease the ranges of all variograms
    """
    vg_list = vg_str.split(' + ')
    sills_list = []
    types_list = []
    ranges_list = []
    for vg in vg_list:
        sill, rst = vg.split(' ')
        vg_type, rst = rst.split('(')
        rng = rst.split(')')[0]
        sills_list.append(float(sill))
        types_list.append(vg_type)
        ranges_list.append(float(rng))

    sum_sills = np.sum(sills_list)
    if sum_sills == 0:
        sill_ratio = 0
    else:
        sill_ratio = new_sill_val / sum_sills
    new_sills_list = np.multiply(sills_list, sill_ratio)

    new_ranges_list = np.multiply(ranges_list, rang_fac)

    new_vg_str = ''
    for i in range(len(sills_list)):
        new_vg_str += '%0.5f ' % new_sills_list[i]
        new_vg_str += types_list[i]
        new_vg_str += '(%0.5f)' % new_ranges_list[i]
        new_vg_str += ' + '

    return new_vg_str[:-3]


def get_ns(ref, sim):
    ''' Get Nash - Sutcliffe score
    '''
    if len(ref.shape) != len(sim.shape) != 1:
        raise Exception(('Shape of ref and sim unequal or the arrays are '
                         'multidimensional...'))

    if np.any(np.isnan(ref)) or np.any(np.isnan(sim)):
        raise Exception('NaNs in the input data...')

    mean_ref = np.mean(ref)
    numr = np.sum(np.square(ref - sim))
    demr = np.sum(np.square(ref - mean_ref))

    ns = 1 - (numr / demr)

    return ns


def get_ln_ns(ref, sim):
    ''' Get ln Nash - Sutcliffe score
    '''
    if len(ref.shape) != len(sim.shape) != 1:
        raise Exception(('Shape of ref and sim unequal or the arrays are '
                         'multidimensional...'))

    if np.any(np.isnan(ref)) or np.any(np.isnan(sim)):
        raise Exception('NaNs in the input data...')

    mean_ref = np.mean(np.log(ref))
    numr = np.sum(np.square(np.log(ref) - np.log(sim)))
    demr = np.sum(np.square(np.log(ref) - mean_ref))

    ln_ns = 1 - (numr / demr)

    return ln_ns


def get_kge(act_arr, sim_arr):
    '''Get Kling-Gupta Efficiency
    '''

    if len(act_arr.shape) != len(sim_arr.shape) != 1:
        raise Exception(('Shape of ref and sim unequal or the arrays are '
                         'multidimensional...'))

    if np.any(np.isnan(act_arr)) or np.any(np.isnan(sim_arr)):
        raise Exception('NaNs in the input data...')

    act_mean = act_arr.mean()
    sim_mean = sim_arr.mean()

    act_std_dev = act_arr.std()
    sim_std_dev = sim_arr.std()

    correl = np.corrcoef(act_arr, sim_arr)[0, 1]

    r = correl
    b = sim_mean / act_mean
    g = (sim_std_dev / sim_mean) / (act_std_dev / act_mean)

    kge = 1 - ((r - 1)**2 + (b - 1)**2 + (g - 1)**2)**0.5
    return kge


def polygonize_raster(in_ras, out_shp, fmt='ESRI Shapefile'):
    '''Transform a given a raster with grids having discrete and
    clustered values (e.g. watersheds) to a vector

    '''
    cmd = 'python "%s" "%s" -f "%s" "%s"' % (polygonize_file, in_ras, fmt, out_shp)
    print(cmd)
    proc = subprocess.Popen(cmd) # print activitities to log_file
    proc.wait()

    return


def deg_min_sec__to__dec_degs(lat, lon, deg_sym, min_sym, sec_sym):

    '''Purpose: To convert degree minute seconds to decimal degrees

    Description of arguments:

    lat, lon (string): latitude and longitude in degrees minutes seconds \
                       as a string.

    deg_sym, min_sym, sec_sym (string): symbols for the degree, minutes \
                                        and seconds respectively. These are \
                                        sometimes not what they appear in the \
                                        file so don't copy them directly, \
                                        just look for the characters that are \
                                        displayed in the python interpreter \
                                        when you print them.'
    '''

    try:

        lat_degs, lat_mins = lat.split(deg_sym)
        lat_mins, lat_secs = lat_mins.split(min_sym)
        lat_secs = lat_secs.split(sec_sym)[0]

        lon_degs, lon_mins = lon.split(deg_sym)
        lon_mins, lon_secs = lon_mins.split(min_sym)
        lon_secs = lon_secs.split(sec_sym)[0]

        lat_dec = float(lat_degs) + (float(lat_mins)/60.) + (float(lat_secs)/3600.)
        lon_dec = float(lon_degs) + (float(lon_mins)/60.) + (float(lon_secs)/3600.)
    except:
        lat_dec, lon_dec = np.nan, np.nan

    return [lat_dec, lon_dec]


def interpolate_lin(x, x_arr, y_arr):
    '''
    Get y given x
        Assuming x_arr is monotonically increasing
    '''

    if np.any(np.where(np.ediff1d(x_arr) < 0, 1, 0)):
        raise RuntimeError('x_arr not monotonically ascending')

    if x <= x_arr[0]:
        return y_arr[0]
    elif x >= x_arr[-1]:
        return y_arr[-1]
    else:
        x_idx_guess = int(((x  - x_arr[0]) / (x_arr[-1] - x_arr[0])) * x_arr.shape[0])
        x_idx_ge_eq = x_idx_guess + 1
        x_idx_le_eq = x_idx_guess

        while not (x_arr[x_idx_le_eq] <= x <= x_arr[x_idx_ge_eq]):
            if x < x_arr[x_idx_le_eq]:
                x_idx_ge_eq -= 1
                x_idx_le_eq -= 1
            else:
                x_idx_ge_eq += 1
                x_idx_le_eq += 1

        y = ((x - x_arr[x_idx_le_eq]) / (x_arr[x_idx_ge_eq] - x_arr[x_idx_le_eq]))
        y *= ((y_arr[x_idx_ge_eq] - y_arr[x_idx_le_eq]))
        y += y_arr[x_idx_le_eq]

        return y

def get_avg_run_time(func, n, vals):
    '''
    run function n times and average return time in seconds
    '''
    start_time = timeit.default_timer()
    for i in range(n):
        func(*vals)
    end_time = timeit.default_timer()

    avg_time = end_time - start_time
    avg_time /= n
    return avg_time


def merge_same_id_shp_poly(in_shp, out_shp, field='DN'):
    '''Merge all polygons with the same ID in the 'field' (from TauDEM)

    Because sometimes there are some polygons from the same catchment,
    this is problem because there can only one cathcment with one ID,
    it is an artifact of gdal_polygonize.
    '''
    cat_ds = ogr.Open(in_shp)
    lyr = cat_ds.GetLayer(0)
    spt_ref = lyr.GetSpatialRef()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_shp):
        driver.DeleteDataSource(out_shp)

    feat_dict = {}
    fid_to_field_dict = {}

    feat = lyr.GetNextFeature()
    while feat:
        fid = feat.GetFID()
        f_val = feat.GetFieldAsString(field)
        feat_dict[fid] = feat.Clone()
        fid_to_field_dict[fid] = f_val
        feat = lyr.GetNextFeature()

    out_ds = driver.CreateDataSource(out_shp)
    out_lyr = out_ds.CreateLayer('0', geom_type=ogr.wkbMultiPolygon)


    in_lyr_dfn = lyr.GetLayerDefn()
    for i in range(0, in_lyr_dfn.GetFieldCount()):
        field_dfn = in_lyr_dfn.GetFieldDefn(i)
        out_lyr.CreateField(field_dfn)
    out_lyr_dfn = out_lyr.GetLayerDefn()

    uniq_vals = np.unique(list(fid_to_field_dict.values()))

    for uniq_val in uniq_vals:
        fid_list = []
        for fid in list(fid_to_field_dict.keys()):
            if fid_to_field_dict[fid] == uniq_val:
                fid_list.append(fid)

        if len(fid_list) > 1:
            cat_feat = feat_dict[fid_list[0]]
            for fid in fid_list[1:]:
                ds_cat = cat_feat.GetGeometryRef()
                curr_cat_feat = feat_dict[fid].Clone()
                curr_cat = curr_cat_feat.GetGeometryRef()

                merged_cat = ds_cat.Union(curr_cat)
                merged_cat_feat = ogr.Feature(out_lyr_dfn)
                merged_cat_feat.SetField(field, uniq_val)
                merged_cat_feat.SetGeometry(merged_cat)
                cat_feat = merged_cat_feat
        else:
            cat = feat_dict[fid_list[0]]
            curr_cat_feat = ogr.Feature(out_lyr_dfn)
            curr_cat_feat.SetField(field, uniq_val)
            curr_cat_feat.SetGeometry(cat.GetGeometryRef())
            cat_feat = curr_cat_feat

        for fid in fid_list:
            del feat_dict[fid]
            del fid_to_field_dict[fid]

        feat_dict[uniq_val] = cat_feat

    assert len(list(feat_dict.keys())) == len(uniq_vals), 'shit happend!'

    for key in list(feat_dict.keys()):
        cat_feat = feat_dict[key]
        out_feat = ogr.Feature(out_lyr_dfn)
        geom = cat_feat.GetGeometryRef()
        out_feat.SetGeometry(geom)
        for i in range(0, out_lyr_dfn.GetFieldCount()):
            out_feat.SetField(out_lyr_dfn.GetFieldDefn(i).GetNameRef(), cat_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)

    out_prj = open((out_shp.rsplit('.',1)[0] + '.prj'), 'w')
    out_prj.write(spt_ref.ExportToWkt())
    out_prj.close()

    cat_ds.Destroy()
    out_ds.Destroy()
    return

def oudin_pet(avg_temp, mon, day_of_mon, latitude, leap_year=False):
    if avg_temp < -5:
        return 0.0

    if mon < 3:
        j_d = (275.0 * (mon / 9.0)) - 30 + day_of_mon
    elif (mon >= 3) and (not leap_year):
        j_d = (275.0 * (mon / 9.0)) - 32 + day_of_mon
    elif (mon >= 3) and leap_year:
        j_d = (275.0 * (mon / 9.0)) - 31 + day_of_mon

    delta = 0.409 * math.sin(((2 * math.pi * j_d) / 365.0) - 1.39)
    omega = math.acos(-math.tan(latitude) * math.tan(delta))

    dr = 1.0 + (0.033 * math.cos((2 * math.pi * j_d) / 365.0))

    re_1 = omega * math.sin(latitude) * math.sin(delta)
    re_2 = math.sin(omega) * math.cos(latitude) * math.cos(delta)
    re = 37.6 * dr * (re_1 + re_2)

    pet = (re / (2.26 * 1000)) * ((avg_temp + 5) * 0.01)
    return pet


def get_ns_var_res(ref, sim, ann_cycle):
    ''' Get Nash - Sutcliffe score

    Instead of the mean, the annual cycle is used
    '''
    if len(ref.shape) != len(sim.shape) != len(ann_cycle) != 1:
        raise Exception(('Shape of ref and sim unequal or the arrays are '
                         'multidimensional...'))

    if (np.any(np.isnan(ref)) or
        np.any(np.isnan(sim)) or
        np.any(np.isnan(ann_cycle))):
        raise Exception('NaNs in the input data...')

    numr = np.sum(np.square(ref - sim))
    demr = np.sum(np.square(ref - ann_cycle))

    ns = 1 - (numr / demr)

    return ns


def get_ln_ns_var_res(ref, sim, ann_cycle):
    ''' Get ln Nash - Sutcliffe score

    Instead of the mean, the annual cycle is used
    '''
    if len(ref.shape) != len(sim.shape) != 1:
        raise Exception(('Shape of ref and sim unequal or the arrays are '
                         'multidimensional...'))

    if (np.any(np.isnan(ref)) or
        np.any(np.isnan(sim)) or
        np.any(np.isnan(ann_cycle))):
        raise Exception('NaNs in the input data...')

    numr = np.sum(np.square(np.log(ref) - np.log(sim)))
    demr = np.sum(np.square(np.log(ref) - np.log(ann_cycle)))

    ln_ns = 1 - (numr / demr)

    return ln_ns
