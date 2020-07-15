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
from collections import OrderedDict
from hbv import HBVCyPy

if __name__ == '__main__':
    print '\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime()
    start = timeit.default_timer() # to get the runtime of the program

    # working directory
    main_dir = r'P:\Synchronize\IWS\HBV_Python_Cython/'

    # input file with temperature, precipitation, PET and discharge
    in_data_file = r'input_4408.csv'

    # the delimiter in the input file and for output the file
    sep = ';'

    # number of initial steps not used in the calculation
        # of the objective function
    warm_up_steps = 182

#    obj_ftn_names = ['ln_ns']
    obj_ftn_names = ['ln_ns', 'ns']
#    obj_ftn_wts = [1.0]
    obj_ftn_wts = [1.0, 4.0]

    cnv_fac = (2107. / 864.)

    # read input file
    os.chdir(main_dir)
    in_data_df = pd.read_csv(in_data_file, sep=sep)

    # specify a dictionary having bounds of each variable
    bounds_dict = OrderedDict()
    bounds_dict['tt_bds'] = (-1, 1)
    bounds_dict['c_melt_bds'] = (0.01, 6)
    bounds_dict['fc_bds'] = (30, 300)
    bounds_dict['beta_bds'] = (1, 6)
    bounds_dict['pwp_bds'] = (25, 300)
    bounds_dict['ur_thresh_bds'] = (0.0, 20.0)
    bounds_dict['k_uu_bds'] = (0.0, 1.0)
    bounds_dict['k_ul_bds'] = (0.005, 1.0)
    bounds_dict['k_d_bds'] = (0.0001, 1.0)
    bounds_dict['k_ll_bds'] = (0.0001, 1.0)

    # get input variable values
    in_data_df.replace(to_replace=np.NaN, value=0.000001, inplace=True)
#    in_data_df = in_data_df.iloc[:300]
    temp_arr_orig = in_data_df['temp'].values
    prec_arr_orig = np.array(in_data_df['prec'].values, dtype='float64')
    pevap_arr = in_data_df['pet'].values
    q_act_arr_orig = in_data_df['discharge'].values

    my_HBVCyPy = HBVCyPy(temp_arr_orig,
                         prec_arr_orig,
                         pevap_arr,
                         q_act_arr_orig,
                         warm_up_steps,
                         bounds_dict,
                         cnv_fac,
                         obj_ftn_names,
                         obj_ftn_wts,
                         test_algo=False,
                         msgs=False)

    #my_HBVCyPy.hbv_mod.tt = -0.178002290245
    #my_HBVCyPy.hbv_mod.c_melt = 3.30964142483
    #my_HBVCyPy.hbv_mod.fc = 99.8788456515
    #my_HBVCyPy.hbv_mod.beta = 1.58749566565
    #my_HBVCyPy.hbv_mod.pwp = 51.2780820323
    #my_HBVCyPy.hbv_mod.ur_thresh = 42.3913883885
    #my_HBVCyPy.hbv_mod.k_uu = 0.695953191344
    #my_HBVCyPy.hbv_mod.k_ul = 0.001
    #my_HBVCyPy.hbv_mod.k_d = 0.998560331273
    #my_HBVCyPy.hbv_mod.k_ll = 0.116796883651
#    my_HBVCyPy.fig_dpi = 300

#    my_HBVCyPy.optimize()
#    print my_HBVCyPy.ns
#    p_arr = my_HBVCyPy._run_sim()
#    c_arr = my_HBVCyPy._run_sim_c()
#    idxs = range(13)
#    for i in idxs:
#        print '\n\n\acolumn no:', i, '\n'
#        for j in range(p_arr.shape[0]):
#            print '%0.5f, %0.5f' % (p_arr[j, i], c_arr[j, i])
#    print my_HBVCyPy.ns
#    my_HBVCyPy.save_simple_opt(os.path.join(os.getcwd(), 'q_err_test_auto_16'), 'q_err_auto', save_de=True)
#    my_HBVCyPy.save_water_bal(os.path.join(os.getcwd(), 'q_err_test_auto_17'), 'q_err_auto', save_de=False)

#    my_HBVCyPy.optimize_kfolds(kfolds=2, nprocs=2)
#    my_HBVCyPy.save_kfolds_opt(os.path.join(os.getcwd(), 'test_de_2folds_name_test'),
#                               'test_de', save_hbv=False, save_de=True, save_wat_bal=True)

#    my_HBVCyPy.debug_mode = True
#    my_HBVCyPy.optimize_multi_pop(nprocs=3, npops=3)
#    my_HBVCyPy.save_pop_opt(os.path.join(os.getcwd(), 'test_de_multi_pop_18'), 'test_de', save_hbv=True)

    my_HBVCyPy.sim_rnd_sims()

    stop = timeit.default_timer()  # Ending time
    print '\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop-start)

