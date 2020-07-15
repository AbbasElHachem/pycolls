# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s

Manual time-based resampling for a given file
"""

import os
import timeit
import time

import pandas as pd

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer() # to get the runtime of the program

#main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep\santa_rs_minerve_prep\data'

main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep\meteo_UNASAM\02_variable_temporal_data'

in_data_file = r'TX_AirTC_Tx_Min.csv'
out_data_file = r'TX_AirTC_Tx_Min_daily.csv'
sep = ';'
in_date_fmt = '%Y-%m-%d %H:%M:%S'
out_date_fmt = '%Y-%m-%d'

os.chdir(main_dir)


def get_nanmean(array):
    nans_bool_arr = pd.np.isnan(array)
    if pd.np.all(nans_bool_arr):
        return pd.np.nan
    else:
        nans_not_bool_arr = pd.np.logical_not(nans_bool_arr)
        n_not_nans = pd.np.where(nans_not_bool_arr)[0].shape[0]
        return round(pd.np.sum(array[nans_not_bool_arr]) / n_not_nans, 2)


in_df = pd.read_csv(in_data_file, index_col=0, sep=sep, encoding='utf-8')
in_df.index = pd.to_datetime(in_df.index, format=in_date_fmt)

in_df = in_df.resample('D').apply(get_nanmean)

# using average values to fill the NaNs for every station
nafill_ser = pd.Series(index=in_df.columns)
nafill_ser[:] = in_df.apply(get_nanmean, axis=0)
in_df = in_df.fillna(value=nafill_ser)

in_df.to_csv(out_data_file, sep=str(sep), date_format=out_date_fmt)

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))

