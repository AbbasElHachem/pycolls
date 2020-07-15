# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

Plot the IDW and Random Mixing fields for the historical data
this also removes top rows and last col from the random mixing fields
"""

import os
import timeit
import time
import shapefile as shp
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib as mpl
from faizpy10 import create_new_dir, list_full_path
from matplotlib.colors import LinearSegmentedColormap

print('\a\a\a\a Started on %s \a\a\a\a' % time.asctime())
start = timeit.default_timer() # to get the runtime of the program

idw_flds_dir = r'D:\Synchronize\Masters_Thesis\hist_ppt_IDW_3rd_run\hist_orig_ASCII_idw_grids_3rd_run__hist_orig_RM_13th_run/'
rm_flds_dir = r'D:\Synchronize\Masters_Thesis\hist_ppt_random_mixing_13th_run\tfm_ppt_fields_per_day/'
avg_rm_flds_dir = r'D:\Synchronize\Masters_Thesis\hist_ppt_random_mixing_13th_run\tfm_ppt_fields/'

#in_cat_shp_loc = r'D:\Synchronize\Masters_Thesis\BW.shp'
in_cat_shp_loc = r'D:\Synchronize\RS_MINERVE\ms_rs_minerve_test_01\vector\upper_neckar_tuebingen.shp'

in_bounds_shp_loc = r'D:\Synchronize\RS_MINERVE\ms_rs_minerve_test_01\vector\upper_neckar_tuebingen_merged_20km_buff.shp'
in_info_loc = r'D:\Synchronize\Masters_Thesis\hist_ppt_random_mixing_13th_run\info.csv'

out_figs_dir = r'D:\Synchronize\Masters_Thesis\thesis_writing\graduate-thesis\Figures\IDW_Rand_Mix_Comparison/'

in_date_fmt_01 = '%Y.%m.%d'
in_date_fmt_02 = '%Y-%m-%d'
date_idx_strt = -14
date_idx_end = -4
sep=';'
fig_size = (8.27, 11.69)
dpi = 150
shp_color = 'k'
strt_date = '18820101'
end_date = '18821231'

create_new_dir(out_figs_dir)

dum_list_01 = list_full_path('.csv', idw_flds_dir)
dum_list_02 = list_full_path('.csv', rm_flds_dir)
dum_list_03 = list_full_path('.csv', avg_rm_flds_dir)

n_dum_01 = len(dum_list_01)
n_dum_02 = len(dum_list_02)
n_dum_03 = len(dum_list_03)

file_df = pd.DataFrame(index=pd.date_range(strt_date, end_date), columns=['idw', 'rm', 'avg_rm'])

for i in range(n_dum_01):

    dum_file_01 = dum_list_01[i]

    date_str_01 = (os.path.basename(dum_file_01)[date_idx_strt:date_idx_end])

    date_1 = dt.datetime.strptime(date_str_01, in_date_fmt_01)

    file_df.loc[date_1]['idw'] = dum_file_01

for j in range(n_dum_02):

    dum_file_02 = dum_list_02[j]

    date_str_02 = (os.path.basename(dum_file_02)[date_idx_strt:date_idx_end])

    date_2 = dt.datetime.strptime(date_str_02, in_date_fmt_02)

    file_df.loc[date_2]['rm'] = dum_file_02

for k in range(n_dum_03):

    dum_file_03 = dum_list_03[k]

    date_str_03 = (os.path.basename(dum_file_03)[date_idx_strt:date_idx_end])

    date_3 = dt.datetime.strptime(date_str_03, in_date_fmt_02)

    file_df.loc[date_3]['avg_rm'] = dum_file_03

file_df.dropna(inplace=True)

idw_flds_list = file_df['idw'].values
rm_flds_list = file_df['rm'].values
avg_rm_flds_list = file_df['avg_rm'].values

info_df = pd.read_csv(in_info_loc, sep=sep, index_col=0)
info_df.index = pd.to_datetime(info_df.index, format='%Y-%m-%d 00:00:00')

bounds_sf = shp.Reader(in_bounds_shp_loc)
x_llim, y_llim, x_ulim, y_ulim = bounds_sf.bbox

sf = shp.Reader(in_cat_shp_loc)
shp_xx = []
shp_yy = []
#plt.figure()
for shape in sf.shapeRecords():
    shp_xx.append([i[0] for i in shape.shape.points[:]])
    shp_yy.append([i[1] for i in shape.shape.points[:]])
#    plt.plot(x, y)
#plt.show(block=False)

if len(idw_flds_list) != len(rm_flds_list):
    raise Exception('Number of idw and random mixing files is not equal.')

old_info_list = []  # for RM
idw_old_info_list = []  # for IDW
avg_rm_old_info_list = []  # for Avg RM

tot_rows, tot_cols = 25, 16
row_span, col_span = 7, 8

#blues = LinearSegmentedColormap.from_list(name='blues', N=15, colors=['white','#00CED1','#800080'])
blues = LinearSegmentedColormap.from_list(name='blues', N=15, colors=['white','#00CED1','#140066'])
plt.register_cmap(cmap=blues)
cmap=plt.get_cmap(blues)
cmap.set_over('#5a0080')

#cmap = plt.get_cmap('Paired')
cmap.set_over('0.25')
cmap.set_under('0.75')

c_bar_ticks = [0.001, 5, 10, 15, 20, 30, 50, 70, 100]

# adjust the color intervals by increasing the number of items
# per list
c_bar_ints = list(np.arange(0, 21, 0.5)) + \
             list(np.arange(20, 61, 0.75)) + \
             list(np.arange(61, 100, 1))

norm = mpl.colors.BoundaryNorm(c_bar_ints, cmap.N)

plt.ioff()
for j in range(len(idw_flds_list)):
    idw_fld_file = idw_flds_list[j]
    rm_fld_file = rm_flds_list[j]
    avg_rm_fld_file = avg_rm_flds_list[j]

    print('Plotting: ', '\n', idw_fld_file, '\n', rm_fld_file, '\n', avg_rm_fld_file, '\n')

    date_str_01 = os.path.basename(idw_fld_file)[date_idx_strt:date_idx_end]
    date_str_02 = os.path.basename(rm_fld_file)[date_idx_strt:date_idx_end]
    date_str_03 = os.path.basename(avg_rm_fld_file)[date_idx_strt:date_idx_end]

    date_1 = dt.datetime.strptime(date_str_01, in_date_fmt_01)
    date_2 = dt.datetime.strptime(date_str_02, in_date_fmt_02)
    date_3 = dt.datetime.strptime(date_str_03, in_date_fmt_02)

    if date_1 != date_2 != date_3:
        raise Exception('Input files are not from the same day')

    plt.figure(figsize=fig_size)

    #==============================================================================
    # Plot IDW
    #==============================================================================
    idw_meta_data = np.genfromtxt(idw_fld_file, max_rows=6)[:, 1]
    idw_cols = int(idw_meta_data[0])
    idw_rows = int(idw_meta_data[1])
    idw_cell_size = idw_meta_data[4]
    idw_no_data_value = idw_meta_data[5]
    idw_x_l_c = idw_meta_data[2]
    idw_y_l_c = idw_meta_data[3]
    idw_x_u_c = idw_x_l_c + float(idw_cols * idw_cell_size)
    idw_y_u_c = idw_y_l_c + float(idw_rows * idw_cell_size)
    idw_data = np.genfromtxt(idw_fld_file, skip_header=6)

    max_ppt = np.nanmax(idw_data)
    min_ppt = np.nanmin(idw_data)

    idw_new_info_list = [idw_cols, idw_rows, idw_cell_size, idw_no_data_value, idw_x_l_c, idw_y_u_c]
    if idw_new_info_list != idw_old_info_list:
        print('metadata not the same')
        idw_x_coords = np.arange(idw_x_l_c, idw_x_u_c * 1.00000001, idw_cell_size)
        idw_y_coords = np.arange(idw_y_l_c, idw_y_u_c * 1.00000001, idw_cell_size)

    idw_old_info_list = [idw_cols, idw_rows, idw_cell_size, idw_no_data_value, idw_x_l_c, idw_y_u_c]

    idw_xx, idw_yy = np.meshgrid(idw_x_coords, idw_y_coords)

    idw_ax = plt.subplot2grid((tot_rows, tot_cols), (0, 0), rowspan=row_span, colspan=col_span)
    idw_ax.pcolormesh(idw_xx, idw_yy, np.flipud(idw_data), cmap=cmap, norm=norm)

    idw_ax.set_xlim(x_llim, x_ulim)
    idw_ax.set_ylim(y_llim, y_ulim)

    for k in range(len(shp_xx)):
        idw_ax.plot(shp_xx[k], shp_yy[k], c=shp_color)

    plt.setp(idw_ax.get_xticklabels(), visible=False)
#    plt.show()
#
#    break

    #==============================================================================
    # Plot Avg random mixing
    #==============================================================================
    avg_rm_meta_data = np.genfromtxt(avg_rm_fld_file, max_rows=6)[:, 1]
    avg_rm_cols = int(avg_rm_meta_data[0])
    avg_rm_rows = int(avg_rm_meta_data[1])
    avg_rm_cell_size = avg_rm_meta_data[4]
    avg_rm_no_data_value = avg_rm_meta_data[5]
    avg_rm_x_l_c = avg_rm_meta_data[2]
    avg_rm_y_l_c = avg_rm_meta_data[3]
    avg_rm_x_u_c = avg_rm_x_l_c + float(avg_rm_cols * avg_rm_cell_size)
    avg_rm_y_u_c = avg_rm_y_l_c + float(avg_rm_rows * avg_rm_cell_size)
    avg_rm_data = np.genfromtxt(avg_rm_fld_file, skip_header=6)

    max_ppt = np.nanmax(avg_rm_data)
    min_ppt = np.nanmin(avg_rm_data)

    avg_rm_new_info_list = [avg_rm_cols, avg_rm_rows, avg_rm_cell_size, avg_rm_no_data_value, avg_rm_x_l_c, avg_rm_y_u_c]
    if avg_rm_new_info_list != avg_rm_old_info_list:
        print('metadata not the same')
        avg_rm_x_coords = np.arange(avg_rm_x_l_c, avg_rm_x_u_c * 1.00000001, avg_rm_cell_size)
        avg_rm_y_coords = np.arange(avg_rm_y_l_c, avg_rm_y_u_c * 1.00000001, avg_rm_cell_size)

    avg_rm_old_info_list = [avg_rm_cols, avg_rm_rows, avg_rm_cell_size, avg_rm_no_data_value, avg_rm_x_l_c, avg_rm_y_u_c]

    avg_rm_xx, avg_rm_yy = np.meshgrid(avg_rm_x_coords, avg_rm_y_coords)

    avg_rm_ax = plt.subplot2grid((tot_rows, tot_cols), (0, col_span), rowspan=row_span, colspan=col_span)
    avg_rm_ax.pcolormesh(avg_rm_xx, avg_rm_yy, np.flipud(avg_rm_data), cmap=cmap, norm=norm)

    avg_rm_ax.set_xlim(x_llim, x_ulim)
    avg_rm_ax.set_ylim(y_llim, y_ulim)

    for k in range(len(shp_xx)):
        avg_rm_ax.plot(shp_xx[k], shp_yy[k], c=shp_color)

    plt.setp(avg_rm_ax.get_xticklabels(), visible=False)
    plt.setp(avg_rm_ax.get_yticklabels(), visible=False)
#    plt.show()
#
#    break
    #==============================================================================
    # Plot Random Mixing
    #==============================================================================
    n_out_dfs = int(info_df.loc[date_1]['n_constr_F'])
    lines_per_field = int(info_df.loc[date_1]['lns_tfm_fld'])

    sel_df_nos = np.unique(np.random.randint(0, n_out_dfs, 10))
#    sel_df_nos = [0]
    sel_df_nos = sel_df_nos[:5]

    row_num, col_num = row_span, 0

    for df_no, line_no in enumerate(range(0, (n_out_dfs * lines_per_field), lines_per_field)):

        if df_no in sel_df_nos:
            if row_num == 0: # dont plot on the avg rm fields
                if row_num  +  2*row_span < tot_rows:
                    row_num = row_num + row_span
                continue

            meta_data = np.genfromtxt(rm_fld_file, skip_header=line_no, max_rows=6)[:, 1]
            cols = int(meta_data[0]) - 1
            rows = int(meta_data[1]) - 1
            cell_size = meta_data[4]
            no_data_value = meta_data[5]
            x_l_c = meta_data[2]
            y_l_c = meta_data[3]
            x_u_c = x_l_c + float(cols * cell_size)
            y_u_c = y_l_c + float(rows * cell_size)
            data = np.genfromtxt(rm_fld_file, skip_header=line_no + 6, max_rows=lines_per_field - 6)[1:, :-1]

            data_max_ppt = np.nanmax(data)
            data_min_ppt = np.nanmin(data)
            if data_min_ppt > min_ppt:
                min_ppt = data_min_ppt

            if data_max_ppt > max_ppt:
                max_ppt = data_max_ppt

            if data.shape[0] != idw_rows or data.shape[1] != idw_cols:
                raise Exception('IDW and Random Mixing rows/cols unequal')

            new_info_list = [cols, rows, cell_size, no_data_value, x_l_c, y_u_c]
            if new_info_list != old_info_list:
                print('metadata not the same')
                x_coords = np.arange(x_l_c, x_u_c * 1.00000001, cell_size)
                y_coords = np.arange(y_l_c, y_u_c * 1.00000001, cell_size)
                xx, yy = np.meshgrid(x_coords, y_coords)

            old_info_list = [cols, rows, cell_size, no_data_value, x_l_c, y_u_c]

            ax = plt.subplot2grid((tot_rows, tot_cols), (row_num, col_num), rowspan=row_span, colspan=col_span, sharex=idw_ax, sharey=idw_ax)
            p_ax = ax.pcolormesh(xx, yy, np.flipud(data), cmap=cmap, norm=norm)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            if col_num > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

            if row_num  +  2*row_span < tot_rows:
                plt.setp(ax.get_xticklabels(), visible=False)
                row_num = row_num + row_span

            elif col_num + col_span < tot_cols:
                col_num = col_num + col_span
                row_num = 0

            if row_num >= tot_rows or col_num >= tot_cols:
                raise Exception('Figures rows or columns exceeded the maximum')

            ax.set_xlim(x_llim, x_ulim)
            ax.set_ylim(y_llim, y_ulim)

            for k in range(len(shp_xx)):
                ax.plot(shp_xx[k], shp_yy[k], c=shp_color)

    ax_l = plt.subplot2grid((tot_rows, tot_cols), (row_num + row_span + 2, 0), rowspan=4, colspan=tot_cols)
    ax_l.set_axis_off()

    cb = plt.colorbar(p_ax, ax=ax_l, fraction=0.75, aspect=20, extend='max', orientation='horizontal')
    cb.set_label('Precipitation (mm)')
    cb.set_ticks(c_bar_ticks)

    plt.savefig(out_figs_dir + 'IDW_RM_Compare_' + date_str_01.replace('.', '-') + '.png', bbox='tight_layout', dpi=dpi)

#    plt.show()
    plt.close()
#    break



stop = timeit.default_timer()  # Ending time
print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start))

