'''
@author: Faizan-Uni-Stuttgart

Template
'''

import os
import timeit
import time
from pathlib import Path

from .core import CodeGenr, write_pyxbld


def write_cp_classi_main_lines(params_dict):
    module_name = 'cp_classi_main'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    out_dir = params_dict['out_dir']

    obj_1_flag = params_dict['obj_1_flag']
    obj_2_flag = params_dict['obj_2_flag']
    obj_3_flag = params_dict['obj_3_flag']
    obj_4_flag = params_dict['obj_4_flag']
    obj_5_flag = params_dict['obj_5_flag']
    obj_6_flag = params_dict['obj_6_flag']
    obj_7_flag = params_dict['obj_7_flag']
    obj_8_flag = params_dict['obj_8_flag']

    op_mp_memb_flag = params_dict['op_mp_memb_flag']
    op_mp_obj_ftn_flag = params_dict['op_mp_obj_ftn_flag']

    pyxcd = CodeGenr(tab=tab)
    pyxbldcd = CodeGenr(tab=tab)

    #==========================================================================
    # add cython flags
    #==========================================================================
    pyxcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pyxcd.w('# cython: boundscheck=%s' % boundscheck)
    pyxcd.w('# cython: wraparound=%s' % str(wraparound))
    pyxcd.w('# cython: cdivision=%s' % str(cdivision))
    pyxcd.w('# cython: language_level=%d' % int(language_level))
    pyxcd.els()

    _ = ';'.join(map(str, [obj_1_flag,
                           obj_2_flag,
                           obj_3_flag,
                           obj_4_flag,
                           obj_5_flag,
                           obj_6_flag,
                           obj_7_flag,
                           obj_8_flag]))
    pyxcd.w('### obj_ftns:' + _)
    pyxcd.els()

    pyxcd.w('### op_mp_memb_flag:' + str(op_mp_memb_flag))
    pyxcd.w('### op_mp_obj_ftn_flag:' + str(op_mp_obj_ftn_flag))
    pyxcd.els()

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    pyxcd.w('from cython.parallel import prange')
    pyxcd.els()
    pyxcd.w(('from .gen_mod_cp_rules cimport '
             '(gen_cp_rules, mod_cp_rules)'))
    pyxcd.w('from .memb_ftns cimport (calc_membs_dof_cps, update_membs_dof_cps)')
    pyxcd.w(('from .cp_obj_ftns cimport '
             '(obj_ftn_refresh, obj_ftn_update)'))
    pyxcd.els()

    #==========================================================================
    # declare types
    #==========================================================================
    pyxcd.w('ctypedef double DT_D')
    pyxcd.w('ctypedef unsigned long DT_UL')
    pyxcd.w('ctypedef long long DT_LL')
    pyxcd.w('ctypedef unsigned long long DT_ULL')
    pyxcd.w('ctypedef np.float64_t DT_D_NP_t')
    pyxcd.w('ctypedef np.uint64_t DT_UL_NP_t')
    pyxcd.els()

    pyxcd.w('DT_D_NP = np.float64')
    pyxcd.w('DT_UL_NP = np.uint64')
    pyxcd.els(2)

    #==========================================================================
    # add external imports
    #==========================================================================
    pyxcd.w('cdef extern from "math.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_D exp(DT_D x)')
    pyxcd.w('bint isnan(DT_D x)')
    pyxcd.ded(lev=2)
    pyxcd.els()

    pyxcd.w('cdef extern from "./rand_gen.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef: ')
    pyxcd.ind()
    pyxcd.w('DT_D rand_c()')
    pyxcd.w('void warm_up()  # call this at least once')
    pyxcd.w('void re_seed(DT_ULL x)  # calls warm_up as well')
    pyxcd.ded(lev=2)
    pyxcd.w('warm_up()')
    pyxcd.els(2)

    #==========================================================================
    # Functions
    #==========================================================================
    pyxcd.w('cpdef classify_cps(dict args_dict):')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('# ulongs')
    pyxcd.w('Py_ssize_t i, j, k, l')
    pyxcd.w('DT_UL n_cps, n_pts, n_time_steps, n_fuzz_nos, n_cpus, msgs, n_max = 0')
    pyxcd.w('DT_UL curr_n_iter, curr_m_iter, max_m_iters, max_n_iters')
    pyxcd.w('DT_UL best_accept_iters, accept_iters, rand_acc_iters, reject_iters')
    pyxcd.w('DT_UL rand_k, rand_i, rand_v, old_v_i_k, run_type, no_cp_val')
    pyxcd.w('DT_UL curr_fuzz_idx, last_best_accept_n_iter, max_idxs_ct')
    pyxcd.w('DT_UL rollback_iters_ct, new_iters_ct, update_iters_ct')
    pyxcd.w('DT_UL max_temp_adj_atmps, curr_temp_adj_iter = 0')
    pyxcd.w('DT_UL max_iters_wo_chng, curr_iters_wo_chng = 0, temp_adjed = 0')
    pyxcd.w('DT_UL temp_adj_iters, min_acc_rate, max_acc_rate')
    pyxcd.els()

    pyxcd.w('# doubles')
    pyxcd.w('DT_D anneal_temp_ini, temp_red_alpha, curr_anneal_temp, p_l')
    pyxcd.w('DT_D best_obj_val, curr_obj_val, pre_obj_val, rand_p, boltz_p')
    pyxcd.w('DT_D acc_rate, temp_inc, lo_freq_pen_wt, min_freq')
    pyxcd.els()

    pyxcd.w('# other variables')
    pyxcd.w('list curr_n_iters_list = []')
    pyxcd.w('list curr_obj_vals_list = []')
    pyxcd.w('list best_obj_vals_list = []')
    pyxcd.w('list acc_rate_list = []')
    pyxcd.w('list cp_pcntge_list = []')
    pyxcd.w('list ants = [[], []]')
    pyxcd.els()

    pyxcd.w('# 1D ulong arrays')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=1, mode=\'c\'] best_sel_cps')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=1, mode=\'c\'] chnge_steps')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=1, mode=\'c\'] sel_cps, old_sel_cps')
    pyxcd.els()

    pyxcd.w('# 2D ulong arrays')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] cp_rules, best_cps')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] cp_rules_idx_ctr')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] best_cp_rules_idx_ctr')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] loc_mod_ctr')
    pyxcd.els()

    pyxcd.w('# 2D double arrays')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] slp_anom, fuzz_nos_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] dofs_arr, best_dofs_arr')
    pyxcd.els()

    pyxcd.w('# 3D double arrays')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] mu_i_k_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] cp_dof_arr')
    pyxcd.els()

    pyxcd.w('# arrays for all obj. ftns.')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] obj_ftn_wts_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] ppt_cp_n_vals_arr')
    pyxcd.els()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] in_ppt_arr')
    pyxcd.els()

    if obj_2_flag or obj_5_flag:
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] in_cats_ppt_arr')
    pyxcd.els()

    if obj_8_flag:
        pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] in_lorenz_arr')
    pyxcd.els()

    if any([obj_1_flag, obj_3_flag, obj_4_flag, obj_2_flag, obj_5_flag]):
        pyxcd.w('# ulongs for obj. ftns.')
    if obj_1_flag or obj_3_flag:
        pyxcd.w('Py_ssize_t m')
        pyxcd.w('DT_UL n_stns')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('Py_ssize_t q')
        pyxcd.w('DT_UL n_cats')

    if obj_8_flag:
        pyxcd.w('Py_ssize_t t')
        pyxcd.w('DT_UL n_lors')

    pyxcd.els()

#     if any([obj_1_flag, obj_3_flag, obj_4_flag, obj_2_flag, obj_5_flag]):
#         pyxcd.w('# doubles for obj. ftns.')
#     pyxcd.els()

    if obj_1_flag:
        pyxcd.w('# ulongs obj. ftn. 1')
        pyxcd.w('Py_ssize_t p')
        pyxcd.w('DT_UL n_o_1_threshs')
        pyxcd.els()
        pyxcd.w('# arrays for obj. ftn. 1')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] o_1_ppt_thresh_arr')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] ppt_mean_pis_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] stns_obj_1_vals_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] ppt_cp_mean_pis_arr')
        pyxcd.els()

    if obj_2_flag:
        pyxcd.w('# doubles obj. ftn. 2')
        pyxcd.w('Py_ssize_t r')
        pyxcd.w('DT_UL n_o_2_threshs')
        pyxcd.els()

        pyxcd.w('# arrays for obj. ftn. 2')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] o_2_ppt_thresh_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] cats_ppt_mean_pis_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] cats_obj_2_vals_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] cats_ppt_cp_mean_pis_arr')
        pyxcd.els()

    if obj_3_flag:
        pyxcd.w('# arrays for obj. ftn. 3')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] ppt_mean_arr')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] ppt_cp_mean_arr')
        pyxcd.els()

    if obj_4_flag or obj_6_flag or obj_7_flag:
        pyxcd.w('# ulongs obj. ftns. 4')
        pyxcd.w('DT_UL n_nebs')
        pyxcd.els()

        pyxcd.w('# arrays for obj. ftns. 4, 6 and 7')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] in_wet_arr_calib')
        pyxcd.els()

    if obj_4_flag:
        pyxcd.w('# ulongs obj. ftn. 4')
        pyxcd.w('Py_ssize_t n, o')
        pyxcd.w('DT_UL n_o_4_threshs')
        pyxcd.els()

        pyxcd.w('# arrays for obj. ftn. 4')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] o_4_p_thresh_arr')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] ppt_mean_wet_arr')

        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] nebs_wet_obj_vals_arr')
        pyxcd.w(
            'np.ndarray[DT_UL_NP_t, ndim=3, mode=\'c\'] ppt_cp_mean_wet_arr')
        pyxcd.els()

    if obj_5_flag:
        pyxcd.w('# arrays for obj. ftn. 5')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] cats_ppt_mean_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] cats_ppt_cp_mean_arr')
        pyxcd.els()

    if obj_6_flag:
        pyxcd.w('DT_D mean_wet_dof = 0.0, min_wettness_thresh')
        pyxcd.els()

        pyxcd.w('# arrays for obj. ftn. 6')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] mean_cp_wet_dof_arr')
        pyxcd.w(
            'np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] wet_dofs_arr')
        pyxcd.els()

    if obj_7_flag:
        pyxcd.w('# for obj. ftn. 7')
        pyxcd.w('DT_D mean_tri_wet = 0.0')

        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] mean_cp_tri_wet_arr')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] tri_wet_arr')
        pyxcd.els()

    if obj_8_flag:
        pyxcd.w('# arrays for obj. ftn. 8')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] mean_lor_arr')
        pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] lor_cp_mean_arr')
        pyxcd.els()

    pyxcd.ded()

    pyxcd.w('# read everythings from the given dict. Must do explicitly.')

    if obj_1_flag or obj_3_flag:
        pyxcd.w('in_ppt_arr = args_dict[\'in_ppt_arr_calib\']')
        pyxcd.w('n_stns = in_ppt_arr.shape[1]')
        pyxcd.w('n_max = max(n_max, n_stns)')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('in_cats_ppt_arr = args_dict[\'in_cats_ppt_arr_calib\']')
        pyxcd.w('n_cats = in_cats_ppt_arr.shape[1]')
        pyxcd.w('n_max = max(n_max, n_cats)')

    if obj_1_flag:
        pyxcd.w('o_1_ppt_thresh_arr = args_dict[\'o_1_ppt_thresh_arr\']')
        pyxcd.w('n_o_1_threshs = o_1_ppt_thresh_arr.shape[0]')

    if obj_2_flag:
        pyxcd.w('o_2_ppt_thresh_arr = args_dict[\'o_2_ppt_thresh_arr\']')
        pyxcd.w('n_o_2_threshs = o_2_ppt_thresh_arr.shape[0]')

    if obj_4_flag or obj_6_flag or obj_7_flag:
        pyxcd.w('in_wet_arr_calib = args_dict[\'in_wet_arr_calib\']')

    if obj_4_flag:
        pyxcd.w('n_nebs = in_wet_arr_calib.shape[1]')
        pyxcd.w('n_max = max(n_max, n_nebs)')
        pyxcd.w('assert n_nebs, \'n_nebs cannot be zero!\'')

    if obj_6_flag:
        pyxcd.w('min_wettness_thresh = args_dict[\'min_wettness_thresh\']')

    if obj_4_flag:
        pyxcd.w('o_4_p_thresh_arr = args_dict[\'o_4_p_thresh_arr\']')
        pyxcd.w('n_o_4_threshs = o_4_p_thresh_arr.shape[0]')

    if obj_8_flag:
        pyxcd.w('in_lorenz_arr = args_dict[\'in_lorenz_arr_calib\']')
        pyxcd.w('n_lors = in_lorenz_arr.shape[1]')
        pyxcd.w('n_max = max(n_max, n_lors)')
        pyxcd.w('assert n_lors, \'n_lors cannot be zero!\'')

    pyxcd.els()
    pyxcd.w('obj_ftn_wts_arr = args_dict[\'obj_ftn_wts_arr\']')
    pyxcd.w('n_cps = args_dict[\'n_cps\']')
    pyxcd.w('no_cp_val = args_dict[\'no_cp_val\']')
    pyxcd.w('p_l = args_dict[\'p_l\']')
    pyxcd.w('fuzz_nos_arr = args_dict[\'fuzz_nos_arr\']')
    pyxcd.w('slp_anom = args_dict[\'slp_anom_calib\']')
    pyxcd.w('anneal_temp_ini = args_dict[\'anneal_temp_ini\']')
    pyxcd.w('temp_red_alpha = args_dict[\'temp_red_alpha\']')
    pyxcd.w('max_m_iters = args_dict[\'max_m_iters\']')
    pyxcd.w('max_n_iters = args_dict[\'max_n_iters\']')
    pyxcd.w('n_cpus = args_dict[\'n_cpus\']')
    pyxcd.w('max_idxs_ct = args_dict[\'max_idxs_ct\']')
    pyxcd.w('max_iters_wo_chng = args_dict[\'max_iters_wo_chng\']')

    pyxcd.w('temp_adj_iters = args_dict[\'temp_adj_iters\']')
    pyxcd.w('min_acc_rate = args_dict[\'min_acc_rate\']')
    pyxcd.w('max_acc_rate = args_dict[\'max_acc_rate\']')
    pyxcd.w('max_temp_adj_atmps = args_dict[\'max_temp_adj_atmps\']')

    pyxcd.w('lo_freq_pen_wt = args_dict[\'lo_freq_pen_wt\']')
    pyxcd.w('min_freq = args_dict[\'min_freq\']')
    pyxcd.els()

    pyxcd.w('if \'msgs\' in args_dict:')
    pyxcd.ind()
    pyxcd.w('msgs = <DT_UL> args_dict[ \'msgs\']')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('msgs = 0')
    pyxcd.ded()

    pyxcd.w('assert n_cps >= 2, \'n_cps cannot be less than 2!\'')
    pyxcd.els()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\n')")
    pyxcd.w('print(\'Calibrating CPs...\')')

    if obj_1_flag or obj_3_flag:
        pyxcd.w('print(\'n_stns:\', n_stns)')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('print(\'n_cats:\', n_cats)')

    if obj_1_flag:
        pyxcd.w('print(\'o_1_ppt_thresh_arr:\', o_1_ppt_thresh_arr)')
        pyxcd.w('print(\'n_o_1_threshs:\', n_o_1_threshs)')

    if obj_2_flag:
        pyxcd.w('print(\'o_2_ppt_thresh_arr:\', o_2_ppt_thresh_arr)')
        pyxcd.w('print(\'n_o_2_threshs:\', n_o_2_threshs)')

    if obj_4_flag:
        pyxcd.w('print(\'n_nebs:\', n_nebs)')

    if obj_4_flag:
        pyxcd.w('print(\'o_4_p_thresh_arr:\', o_4_p_thresh_arr)')
        pyxcd.w('print(\'n_o_4_threshs:\', n_o_4_threshs)')

    if obj_4_flag or obj_6_flag or obj_7_flag:
        pyxcd.w('print(\'in_wet_arr shape:\', (in_wet_arr_calib.shape[0], in_wet_arr_calib.shape[1]))')

    if obj_6_flag:
        pyxcd.w('print(\'min_wettness_thresh:\', min_wettness_thresh)')

    if obj_8_flag:
        pyxcd.w('print(\'in_lorenz_arr shape:\', (in_lorenz_arr.shape[0], in_lorenz_arr.shape[1]))')

    pyxcd.w('print(\'n_cps:\', n_cps)')
    pyxcd.w('print(\'n_cpus:\', n_cpus)')
    pyxcd.w('print(\'no_cp_val:\', no_cp_val)')
    pyxcd.w('print(\'p_l:\', p_l)')
    pyxcd.w(r"print('fuzz_nos_arr:\n', fuzz_nos_arr)")
    pyxcd.w('print(\'anneal_temp_ini:\', anneal_temp_ini)')
    pyxcd.w('print(\'temp_red_alpha:\', temp_red_alpha)')
    pyxcd.w('print(\'max_m_iters:\', max_m_iters)')
    pyxcd.w('print(\'max_n_iters:\', max_n_iters)')
    pyxcd.w('print(\'max_idxs_ct:\', max_idxs_ct)')
    pyxcd.w('print(\'obj_ftn_wts_arr:\', obj_ftn_wts_arr)')
    pyxcd.w('print(\'anom shape: (%d, %d)\' % '
            '(slp_anom.shape[0], slp_anom.shape[1]))')

    pyxcd.w('print(\'max_iters_wo_chng:\', max_iters_wo_chng)')
    pyxcd.w('print(\'temp_adj_iters:\', temp_adj_iters)')
    pyxcd.w('print(\'min_acc_rate:\', min_acc_rate)')
    pyxcd.w('print(\'max_acc_rate:\', max_acc_rate)')
    pyxcd.w('print(\'max_temp_adj_atmps:\', max_temp_adj_atmps)')

    pyxcd.w('print(\'lo_freq_pen_wt:\', lo_freq_pen_wt)')
    pyxcd.w('print(\'min_freq:\', min_freq)')
    pyxcd.w('print(\'n_max:\', n_max)')

    if obj_1_flag or obj_3_flag:
        pyxcd.w('print(\'in_ppt_arr shape: (%d, %d)\' % '
                '(in_ppt_arr.shape[0], in_ppt_arr.shape[1]))')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('print(\'in_cats_ppt_arr shape: (%d, %d)\' % '
                '(in_cats_ppt_arr.shape[0], in_cats_ppt_arr.shape[1]))')

    pyxcd.ded()

    pyxcd.w('# initialize the required variables')
    pyxcd.w('n_pts = slp_anom.shape[1]')
    pyxcd.w('n_fuzz_nos = fuzz_nos_arr.shape[0]')
    pyxcd.w('n_time_steps = slp_anom.shape[0]')
    pyxcd.els()

    pyxcd.w('if max_idxs_ct > (n_pts / n_fuzz_nos):')
    pyxcd.ind()
    pyxcd.w('max_idxs_ct = <DT_UL> (n_pts / n_fuzz_nos)')
    pyxcd.w(r'print(("\n\n\n\n######### max_idxs_ct reset to %d!#########\n\n\n\n" % max_idxs_ct))')
    pyxcd.ded()
    pyxcd.w('curr_n_iter = 0')
    pyxcd.w('curr_m_iter = 0')
    pyxcd.w('curr_iters_wo_chng = 0')
    pyxcd.w('best_obj_val = -np.inf')
    pyxcd.w('pre_obj_val = best_obj_val')
    pyxcd.w('curr_anneal_temp = anneal_temp_ini  # to change temp on the fly')
    pyxcd.els()

    pyxcd.w('best_accept_iters = 0')
    pyxcd.w('accept_iters = 0')
    pyxcd.w('rand_acc_iters = 0')
    pyxcd.w('reject_iters = 0')
    pyxcd.els()

    pyxcd.w('new_iters_ct = 0')
    pyxcd.w('update_iters_ct = 0')
    pyxcd.w('rollback_iters_ct = 0')
    pyxcd.els()

    pyxcd.w('rand_k = n_cps - 1')
    pyxcd.w('rand_i = n_pts - 1')
    pyxcd.w('rand_v = n_fuzz_nos')
    pyxcd.w('old_v_i_k = n_fuzz_nos')
    pyxcd.els()

    pyxcd.els()

    pyxcd.w('# run_type == 1 means fresh start i.e. everything is reset '
            'except')
    pyxcd.w('# the cp_rules. This is done when curr_m_iter >= '
            'max_m_iter as well.')
    pyxcd.w('# runtype == 2 means an update cycle i.e. values at the '
            'given CP and point')
    pyxcd.w('# are changed. Also, only those days that have a changed CP '
            'are evaluated')
    pyxcd.w('# further.')
    pyxcd.w('# run_type = 3 means a rollback cycle i.e. everything is set '
            'to the last')
    pyxcd.w('# value that it had on successful run.')
    pyxcd.w('run_type = 1')
    pyxcd.els()

    pyxcd.w('# initialize the required arrays')
    pyxcd.w('cp_rules = np.random.randint(0, '
            'n_fuzz_nos + 1, size=(n_cps, n_pts), '
            'dtype=DT_UL_NP)')

    pyxcd.els()
    pyxcd.w('cp_rules_idx_ctr = np.zeros(shape=(n_cps, n_fuzz_nos), '
            'dtype=DT_UL_NP)')
    pyxcd.w('best_cp_rules_idx_ctr = cp_rules_idx_ctr.copy()')
    pyxcd.w('loc_mod_ctr = np.zeros((n_cps, n_pts), dtype=DT_UL_NP)')
    pyxcd.els()

    pyxcd.w('gen_cp_rules(')
    pyxcd.ind()
    pyxcd.w('cp_rules,')
    pyxcd.w('cp_rules_idx_ctr,')
    pyxcd.w('max_idxs_ct,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_pts,')
    pyxcd.w('n_fuzz_nos,')
    pyxcd.w('n_cpus)')
    pyxcd.ded()
    pyxcd.w('best_cps = cp_rules.copy()')
    pyxcd.w('best_sel_cps = np.zeros(n_time_steps, dtype=DT_UL_NP)')
    pyxcd.els()

    pyxcd.w('uni_cps, cps_freqs = np.unique(best_sel_cps, return_counts=True)')
    pyxcd.w('cp_rel_freqs = 100 * cps_freqs / float(n_time_steps)')
    pyxcd.w('cp_rel_freqs = np.round(cp_rel_freqs, 2)')
    pyxcd.els()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\n%-10s:%s' % ('Unique CPs', 'Relative Frequencies (%)'))")
    pyxcd.w('for x, y in zip(uni_cps, cp_rel_freqs):')
    pyxcd.ind()
    pyxcd.w('print(\'%10d:%-20.2f\' % (x, y))')

    pyxcd.ded()

    pyxcd.w(r"print('\ncp_rules_idx_ctr:\n', cp_rules_idx_ctr.T)")
    pyxcd.w(r"print('\nbest_cp_rules_idx_ctr:\n', best_cp_rules_idx_ctr.T)")
    pyxcd.w(r"print(50 * '#', '\n\n')")
    pyxcd.ded()

    pyxcd.w('mu_i_k_arr = np.zeros(shape=(n_time_steps, n_cps, n_pts), '
            'dtype=DT_D_NP)')
    pyxcd.w('cp_dof_arr = np.zeros(shape=(n_time_steps, n_cps, n_fuzz_nos), '
            'dtype=DT_D_NP)')
    pyxcd.w('sel_cps = np.full(n_time_steps, no_cp_val, dtype=DT_UL_NP)')
    pyxcd.w('old_sel_cps = sel_cps.copy()')
    pyxcd.els()

    pyxcd.w('chnge_steps = np.zeros(n_time_steps, dtype=DT_UL_NP)')
    pyxcd.w('dofs_arr = np.full((n_time_steps, n_cps), 0.0, dtype=DT_D_NP)')
    pyxcd.w('best_dofs_arr = dofs_arr.copy()')
    pyxcd.els()

    pyxcd.w('# initialize the obj. ftn. variables')
    pyxcd.w('ppt_cp_n_vals_arr = np.full(n_cps, 0.0, dtype=DT_D_NP)')
    pyxcd.els()

    if obj_1_flag:
        pyxcd.w('# initialize obj. ftn. 1 variables')
        pyxcd.w(
            'ppt_mean_pis_arr = np.full((n_stns, n_o_1_threshs), 0.0, dtype=DT_D_NP)')
        pyxcd.w(
            'ppt_cp_mean_pis_arr = np.full((n_stns, n_cps, n_o_1_threshs), 0.0, dtype=DT_D_NP)')
        pyxcd.w('stns_obj_1_vals_arr = np.full((n_stns, n_o_1_threshs), '
                '0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_2_flag:
        pyxcd.w('# initialize obj. ftn. 2 variables')
        pyxcd.w('cats_ppt_mean_pis_arr = np.full((n_cats, n_o_2_threshs), '
                '0.0, dtype=DT_D_NP)')
        pyxcd.w('cats_ppt_cp_mean_pis_arr = np.full((n_cats, n_cps, '
                'n_o_2_threshs), 0.0, dtype=DT_D_NP)')
        pyxcd.w('cats_obj_2_vals_arr = np.full((n_cats, n_o_2_threshs), '
                '0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_3_flag:
        pyxcd.w('# initialize obj. ftn. 3 variables')
        pyxcd.w('ppt_mean_arr = np.full(n_stns, 0.0, dtype=DT_D_NP)')
        pyxcd.w('ppt_cp_mean_arr = np.full((n_cps, n_stns), 0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_4_flag:
        pyxcd.w('# initialize obj. ftn. 4 variables')
        pyxcd.w('ppt_mean_wet_arr = np.full((n_nebs, n_o_4_threshs), '
                '0.0, dtype=DT_D_NP)')
        pyxcd.w('ppt_cp_mean_wet_arr = np.full((n_nebs, n_cps, '
                'n_o_4_threshs), 0.0, dtype=DT_UL_NP)')
        pyxcd.w('nebs_wet_obj_vals_arr = np.full((n_nebs, n_o_4_threshs), '
                '0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_5_flag:
        pyxcd.w('# initialize obj. ftn. 5 variables')
        pyxcd.w('cats_ppt_mean_arr = np.full(n_cats, 0.0, dtype=DT_D_NP)')
        pyxcd.w('cats_ppt_cp_mean_arr = np.full((n_cps, n_cats), 0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_6_flag:
        pyxcd.w('# initialize obj. ftn. 6 variables')
        pyxcd.w('mean_cp_wet_dof_arr = np.full(n_cps, '
                '0.0, dtype=DT_D_NP)')
        pyxcd.w('wet_dofs_arr = np.full(n_time_steps, '
                '0.0, dtype=DT_D_NP)')

        pyxcd.els()

    if obj_7_flag:
        pyxcd.w('# initialize obj. ftn. 6 variables')
        pyxcd.w('mean_cp_tri_wet_arr = np.full(n_cps, '
                '0.0, dtype=DT_D_NP)')
        pyxcd.w('tri_wet_arr = np.full(n_time_steps, '
                '0.0, dtype=DT_D_NP)')

        pyxcd.els()

    if obj_8_flag:
        pyxcd.w('# initialize obj. ftn. 8 variables')
        pyxcd.w('mean_lor_arr = np.full(n_lors, 0.0, dtype=DT_D_NP)')
        pyxcd.w('lor_cp_mean_arr = np.full((n_cps, n_lors), 0.0, dtype=DT_D_NP)')
        pyxcd.els()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('# fill some arrays used for obj. 1 and 3 ftns.')
        pyxcd.w('for m in range(n_stns):')
        pyxcd.ind()

        if obj_1_flag:
            pyxcd.w('for p in range(n_o_1_threshs):')
            pyxcd.ind()
            pyxcd.w('ppt_mean_pis_arr[m, p] = np.mean(in_ppt_arr[:, m] > '
                    'o_1_ppt_thresh_arr[p])')
            pyxcd.w(
                'assert (not isnan(ppt_mean_pis_arr[m, p]) and (ppt_mean_pis_arr[m, p] > 0))')
            pyxcd.ded()

        if obj_3_flag:
            pyxcd.w('ppt_mean_arr[m] = np.mean(in_ppt_arr[:, m])')
            pyxcd.w(
                'assert ((not isnan(ppt_mean_arr[m])) and (ppt_mean_arr[m]> 0))')

        pyxcd.ded()

    if obj_2_flag or obj_5_flag:
        pyxcd.w('# fill some arrays used for obj. 2 and 5 ftns.')
        pyxcd.w('for q in range(n_cats):')
        pyxcd.ind()

        if obj_2_flag:
            pyxcd.w('for r in range(n_o_2_threshs):')
            pyxcd.ind()
            pyxcd.w('cats_ppt_mean_pis_arr[q, r] = np.mean(in_cats_ppt_arr[:, q] > '
                    'o_2_ppt_thresh_arr[r])')
            pyxcd.w('assert (not isnan(cats_ppt_mean_pis_arr[q, r]) and '
                    '(cats_ppt_mean_pis_arr[q, r] > 0))')
            pyxcd.ded()

        if obj_5_flag:
            pyxcd.w('cats_ppt_mean_arr[q] = np.mean(in_cats_ppt_arr[:, q])')
            pyxcd.w(
                'assert ((not isnan(cats_ppt_mean_arr[q])) and (cats_ppt_mean_arr[q]> 0))')

        pyxcd.ded()

    if obj_4_flag:
        pyxcd.w('# fill some arrays used for obj. 4 ftns.')
        pyxcd.w('for n in range(n_nebs):')
        pyxcd.ind()
        pyxcd.w('for o in range(n_o_4_threshs):')
        pyxcd.ind()
        pyxcd.w('ppt_mean_wet_arr[n, o] = np.mean(in_wet_arr_calib[:, n] > '
                'o_4_p_thresh_arr[o])')
        pyxcd.w('assert (not isnan(ppt_mean_wet_arr[n, o]))')
        pyxcd.ded(lev=2)

    if obj_6_flag:
        pyxcd.w('# obj. 6 ftns.')
        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('wet_dofs_arr[i] = in_wet_arr_calib[i, 0] + '
                'in_wet_arr_calib[i, 1] - (2 * in_wet_arr_calib[i, 0] * '
                'in_wet_arr_calib[i, 1])')

        pyxcd.ded()
        pyxcd.w('wet_dofs_arr[wet_dofs_arr < min_wettness_thresh] = 0.0')
        pyxcd.w('mean_wet_dof = wet_dofs_arr.mean()')
        pyxcd.w('assert ((not isnan(mean_wet_dof)) and (mean_wet_dof > 0))')
        pyxcd.els()

    if obj_7_flag:
        pyxcd.w('# obj. 7 ftns.')
        pyxcd.w('for i in range(n_time_steps):')
        pyxcd.ind()
        pyxcd.w('tri_wet_arr[i] += np.sum(in_wet_arr_calib[i, :])')
        pyxcd.w('tri_wet_arr[i] += in_wet_arr_calib[i, 0] + in_wet_arr_calib[i, 1] - in_wet_arr_calib[i, 2] + 1')
        pyxcd.w('tri_wet_arr[i] += in_wet_arr_calib[i, 1] + in_wet_arr_calib[i, 2] - in_wet_arr_calib[i, 0] + 1')
        pyxcd.w('tri_wet_arr[i] += in_wet_arr_calib[i, 0] + in_wet_arr_calib[i, 2] - in_wet_arr_calib[i, 1] + 1')

        pyxcd.ded()
        pyxcd.w('mean_tri_wet = tri_wet_arr.mean()')
        pyxcd.w('assert ((not isnan(mean_tri_wet)) and (mean_tri_wet > 0))')
        pyxcd.els()

    if obj_8_flag:
        pyxcd.w('# fill some arrays used for obj. 8 ftn.')
        pyxcd.w('for t in range(n_lors):')
        pyxcd.ind()
        pyxcd.w('mean_lor_arr[t] = np.mean(in_lorenz_arr[:, t])')
        pyxcd.w(
            'assert ((not isnan(mean_lor_arr[t])) and (mean_lor_arr[t] > 0))')

        pyxcd.ded()

    pyxcd.w('# start simulated annealing')
    pyxcd.w('while ((curr_n_iter < max_n_iters) and '
            '(curr_iters_wo_chng < max_iters_wo_chng)) or '
            '(not temp_adjed):')
    pyxcd.ind()

    pyxcd.w('if (curr_m_iter >= max_m_iters) and '
            '(run_type == 2) and '
            '(temp_adjed):')
    pyxcd.ind()
    pyxcd.w('curr_m_iter = 0')
    pyxcd.w('curr_anneal_temp *= temp_red_alpha')
    pyxcd.w('run_type = 1')
    pyxcd.ded()

    pyxcd.w('mod_cp_rules(')
    pyxcd.ind()
    pyxcd.w('cp_rules,')
    pyxcd.w('cp_rules_idx_ctr,')
    pyxcd.w('loc_mod_ctr,')
    pyxcd.w('max_idxs_ct,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_pts,')
    pyxcd.w('n_fuzz_nos,')
    pyxcd.w('run_type,')
    pyxcd.w('&rand_k,')
    pyxcd.w('&rand_i,')
    pyxcd.w('&rand_v,')
    pyxcd.w('&old_v_i_k)')
    pyxcd.ded()

    pyxcd.w('if run_type == 1:')
    pyxcd.ind()
#     pyxcd.w('for i in prange(n_time_steps, nogil=True, schedule=\'dynamic\', '
#             'num_threads=n_cpus):')
#     pyxcd.ind()
#     pyxcd.w('for j in range(n_cps):')
#     pyxcd.ind()
#     pyxcd.w('for k in range(n_pts):')
#     pyxcd.ind()
#     pyxcd.w('mu_i_k_arr[i, j, k] = 0.0')
#     pyxcd.ded()
#     pyxcd.w('dofs_arr[i, j] = 0.0')
#     pyxcd.ded()
#
#     pyxcd.w('sel_cps[i] = 0')
#     pyxcd.w('old_sel_cps[i] = no_cp_val')
#     pyxcd.w('chnge_steps[i] = 0')
#     pyxcd.ded()

    pyxcd.w('new_iters_ct += 1')
    pyxcd.ded()

    pyxcd.w('elif run_type == 2:')
    pyxcd.ind()
    pyxcd.w('update_iters_ct += 1')
    pyxcd.ded()

    pyxcd.w('elif run_type == 3:')
    pyxcd.ind()
    pyxcd.w('rollback_iters_ct += 1')
    pyxcd.ded()

    pyxcd.w('# fill/update the membership, DOF and selected CPs arrays')
    pyxcd.w('if run_type == 1:')
    pyxcd.ind()
    pyxcd.w('calc_membs_dof_cps(')
    pyxcd.ind()
    pyxcd.w('cp_rules,')
    pyxcd.w('mu_i_k_arr,')
    pyxcd.w('cp_dof_arr,')
    pyxcd.w('slp_anom,')
    pyxcd.w('fuzz_nos_arr,')
    pyxcd.w('dofs_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('old_sel_cps,')
    pyxcd.w('chnge_steps,')
    pyxcd.w('no_cp_val,')
    pyxcd.w('p_l,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_time_steps,')
    pyxcd.w('n_pts,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_fuzz_nos)')
    pyxcd.ded(lev=2)

    pyxcd.w('elif run_type == 2:')
    pyxcd.ind()
    pyxcd.w('update_membs_dof_cps(')
    pyxcd.ind()
    pyxcd.w('old_v_i_k,')
    pyxcd.w('rand_v,')
    pyxcd.w('rand_k,')
    pyxcd.w('rand_i,')
    pyxcd.w('cp_rules,')
    pyxcd.w('mu_i_k_arr,')
    pyxcd.w('cp_dof_arr,')
    pyxcd.w('cp_rules_idx_ctr,')
    pyxcd.w('slp_anom,')
    pyxcd.w('fuzz_nos_arr,')
    pyxcd.w('dofs_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('old_sel_cps,')
    pyxcd.w('chnge_steps,')
    pyxcd.w('no_cp_val,')
    pyxcd.w('p_l,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_time_steps,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_fuzz_nos)')
    pyxcd.ded(lev=2)

    pyxcd.w('elif run_type == 3:')
    pyxcd.ind()
    pyxcd.w('update_membs_dof_cps(')
    pyxcd.ind()
    pyxcd.w('rand_v,')
    pyxcd.w('old_v_i_k,')
    pyxcd.w('rand_k,')
    pyxcd.w('rand_i,')
    pyxcd.w('cp_rules,')
    pyxcd.w('mu_i_k_arr,')
    pyxcd.w('cp_dof_arr,')
    pyxcd.w('cp_rules_idx_ctr,')
    pyxcd.w('slp_anom,')
    pyxcd.w('fuzz_nos_arr,')
    pyxcd.w('dofs_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('old_sel_cps,')
    pyxcd.w('chnge_steps,')
    pyxcd.w('no_cp_val,')
    pyxcd.w('p_l,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_time_steps,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_fuzz_nos)')
    pyxcd.ded(lev=2)

    pyxcd.w('# calculate the objective function values')
    pyxcd.w('if run_type == 1:')
    pyxcd.ind()
    pyxcd.w('# start from the begining')
    pyxcd.w('curr_obj_val = obj_ftn_refresh(')
    pyxcd.ind()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('in_ppt_arr,')
        pyxcd.w('n_stns,')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('in_cats_ppt_arr,')
        pyxcd.w('n_cats,')

    if obj_1_flag:
        pyxcd.w('ppt_cp_mean_pis_arr,')
        pyxcd.w('ppt_mean_pis_arr,')
        pyxcd.w('o_1_ppt_thresh_arr,')
        pyxcd.w('stns_obj_1_vals_arr,')
        pyxcd.w('n_o_1_threshs,')

    if obj_2_flag:
        pyxcd.w('cats_ppt_cp_mean_pis_arr,')
        pyxcd.w('cats_ppt_mean_pis_arr,')
        pyxcd.w('o_2_ppt_thresh_arr,')
        pyxcd.w('cats_obj_2_vals_arr,')
        pyxcd.w('n_o_2_threshs,')

    if obj_3_flag:
        pyxcd.w('ppt_cp_mean_arr,')
        pyxcd.w('ppt_mean_arr,')

    if obj_4_flag:
        pyxcd.w('in_wet_arr_calib,')
        pyxcd.w('ppt_mean_wet_arr,')
        pyxcd.w('o_4_p_thresh_arr,')
        pyxcd.w('ppt_cp_mean_wet_arr,')
        pyxcd.w('nebs_wet_obj_vals_arr,')
        pyxcd.w('n_o_4_threshs,')
        pyxcd.w('n_nebs,')

    if obj_5_flag:
        pyxcd.w('cats_ppt_cp_mean_arr,')
        pyxcd.w('cats_ppt_mean_arr,')

    if obj_6_flag:
        pyxcd.w('mean_wet_dof,')
        pyxcd.w('mean_cp_wet_dof_arr,')
        pyxcd.w('wet_dofs_arr,')

    if obj_7_flag:
        pyxcd.w('mean_tri_wet,')
        pyxcd.w('mean_cp_tri_wet_arr,')
        pyxcd.w('tri_wet_arr,')

    if obj_8_flag:
        pyxcd.w('in_lorenz_arr,')
        pyxcd.w('mean_lor_arr,')
        pyxcd.w('lor_cp_mean_arr,')
        pyxcd.w('n_lors,')

    pyxcd.w('ppt_cp_n_vals_arr,')
    pyxcd.w('obj_ftn_wts_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('lo_freq_pen_wt,')
    pyxcd.w('min_freq,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_max,')
    pyxcd.w('n_time_steps,')

    pyxcd.w(')')

    pyxcd.ded()

    pyxcd.w('run_type = 2')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('# only update at steps where the CP has changed')
    pyxcd.w('curr_obj_val = obj_ftn_update(')
    pyxcd.ind()

    if obj_1_flag or obj_3_flag:
        pyxcd.w('in_ppt_arr,')
        pyxcd.w('n_stns,')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('in_cats_ppt_arr,')
        pyxcd.w('n_cats,')

    if obj_1_flag:
        pyxcd.w('ppt_cp_mean_pis_arr,')
        pyxcd.w('ppt_mean_pis_arr,')
        pyxcd.w('o_1_ppt_thresh_arr,')
        pyxcd.w('stns_obj_1_vals_arr,')
        pyxcd.w('n_o_1_threshs,')

    if obj_2_flag:
        pyxcd.w('cats_ppt_cp_mean_pis_arr,')
        pyxcd.w('cats_ppt_mean_pis_arr,')
        pyxcd.w('o_2_ppt_thresh_arr,')
        pyxcd.w('cats_obj_2_vals_arr,')
        pyxcd.w('n_o_2_threshs,')

    if obj_3_flag:
        pyxcd.w('ppt_cp_mean_arr,')
        pyxcd.w('ppt_mean_arr,')

    if obj_4_flag:
        pyxcd.w('in_wet_arr_calib,')
        pyxcd.w('ppt_mean_wet_arr,')
        pyxcd.w('o_4_p_thresh_arr,')
        pyxcd.w('ppt_cp_mean_wet_arr,')
        pyxcd.w('nebs_wet_obj_vals_arr,')
        pyxcd.w('n_o_4_threshs,')
        pyxcd.w('n_nebs,')

    if obj_5_flag:
        pyxcd.w('cats_ppt_cp_mean_arr,')
        pyxcd.w('cats_ppt_mean_arr,')

    if obj_6_flag:
        pyxcd.w('mean_wet_dof,')
        pyxcd.w('mean_cp_wet_dof_arr,')
        pyxcd.w('wet_dofs_arr,')

    if obj_7_flag:
        pyxcd.w('mean_tri_wet,')
        pyxcd.w('mean_cp_tri_wet_arr,')
        pyxcd.w('tri_wet_arr,')

    if obj_8_flag:
        pyxcd.w('in_lorenz_arr,')
        pyxcd.w('mean_lor_arr,')
        pyxcd.w('lor_cp_mean_arr,')
        pyxcd.w('n_lors,')

    pyxcd.w('ppt_cp_n_vals_arr,')
    pyxcd.w('obj_ftn_wts_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('old_sel_cps,')
    pyxcd.w('chnge_steps,')
    pyxcd.w('lo_freq_pen_wt,')
    pyxcd.w('min_freq,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_max,')
    pyxcd.w('n_time_steps,')
    pyxcd.w(')')
    pyxcd.ded(lev=2)

    pyxcd.w('#print(curr_m_iter, curr_n_iter, run_type, round(curr_obj_val, 2), round(pre_obj_val, 2))')

    pyxcd.w('if run_type == 3:')
    pyxcd.ind()
    pyxcd.w('run_type = 2')

    pyxcd.w('for i in range(n_time_steps):')
    pyxcd.ind()
    pyxcd.w('old_sel_cps[i] = sel_cps[i]')
    pyxcd.ded()

    pyxcd.w('continue')
    pyxcd.ded()

    pyxcd.w('assert not isnan(curr_obj_val), \'curr_obj_val is NaN!(%s)\' % curr_n_iter')
    pyxcd.els()

    pyxcd.w('#print(curr_m_iter, curr_n_iter, run_type, round(curr_obj_val, 2), round(pre_obj_val, 2))')
    pyxcd.els()

    pyxcd.w('# a maximizing function')
    pyxcd.w('if (curr_obj_val > best_obj_val) and (run_type == 2):')
    pyxcd.ind()
    pyxcd.w('best_obj_val = curr_obj_val')
    pyxcd.w('last_best_accept_n_iter = curr_n_iter')
    pyxcd.w('for i in range(n_time_steps):')
    pyxcd.ind()
    pyxcd.w('best_sel_cps[i] = sel_cps[i]')
    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('best_dofs_arr[i, j] = dofs_arr[i, j]')
    pyxcd.ded(lev=2)

    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('for k in range(n_pts):')
    pyxcd.ind()
    pyxcd.w('best_cps[j, k] = cp_rules[j, k]')
    pyxcd.ded()
    pyxcd.w('for l in range(n_fuzz_nos):')
    pyxcd.ind()
    pyxcd.w('best_cp_rules_idx_ctr[j, l] = cp_rules_idx_ctr[j, l]')
    pyxcd.ded(lev=2)

    pyxcd.w('best_accept_iters += 1')
    pyxcd.ded()

    pyxcd.w('if curr_obj_val > pre_obj_val:')
    pyxcd.ind()
    pyxcd.w('pre_obj_val = curr_obj_val')
    pyxcd.w('accept_iters += 1')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('rand_p = rand_c()')
    pyxcd.w('boltz_p = exp((curr_obj_val - pre_obj_val) / curr_anneal_temp)')
    pyxcd.w('if rand_p < boltz_p:')
    pyxcd.ind()
    pyxcd.w('pre_obj_val = curr_obj_val')
    pyxcd.w('rand_acc_iters += 1')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('run_type = 3')
    pyxcd.w('#cp_rules[rand_k, rand_i] = old_v_i_k')
    pyxcd.w('reject_iters += 1')
    pyxcd.ded(lev=2)

    pyxcd.w('if run_type == 3:')
    pyxcd.ind()
    pyxcd.w('curr_iters_wo_chng += 1')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('curr_iters_wo_chng = 0')
    pyxcd.ded()

    pyxcd.w('acc_rate = round(100.0 * (accept_iters + rand_acc_iters) / '
            '(accept_iters + rand_acc_iters + reject_iters), 6)')
    pyxcd.els()

    pyxcd.w('if (not curr_m_iter) and temp_adjed:')
    pyxcd.ind()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\ncurr_m_iter:', curr_m_iter)")
    pyxcd.w('print(\'curr_n_iter:\', curr_n_iter)')
    pyxcd.els()

    pyxcd.w('print(\'curr_obj_val:\', curr_obj_val)')
    pyxcd.w('print(\'pre_obj_val:\', pre_obj_val)')
    pyxcd.w('print(\'best_obj_val:\', best_obj_val)')
    pyxcd.els()

    pyxcd.w('print(\'best_accept_iters:\', best_accept_iters)')
    pyxcd.w('print(\'last_best_accept_n_iter:\', last_best_accept_n_iter)')
    pyxcd.w('print(\'accept_iters:\', accept_iters)')
    pyxcd.w('print(\'rand_acc_iters:\', rand_acc_iters)')
    pyxcd.w('print(\'reject_iters:\', reject_iters)')
    pyxcd.w('print(\'curr_anneal_temp:\', curr_anneal_temp)')
    pyxcd.els()

    pyxcd.w('print(\'new_iters_ct:\', new_iters_ct)')
    pyxcd.w('print(\'update_iters_ct:\', update_iters_ct)')
    pyxcd.w('print(\'rollback_iters_ct:\', rollback_iters_ct)')
    pyxcd.els()

    pyxcd.w('print(\'acceptance rate (%age):\', acc_rate)')
    pyxcd.w('print(\'curr_iters_wo_chng:\', curr_iters_wo_chng)')
    pyxcd.els()

    pyxcd.w('#print(\'cp_dof_arr min, max:\', cp_dof_arr.min(), cp_dof_arr.max())')
    pyxcd.els()
    pyxcd.w('print(\'rand_p, boltz_p:\', rand_p, boltz_p)')

    pyxcd.w('uni_cps, cps_freqs = np.unique(best_sel_cps, return_counts=True)')
    pyxcd.w('cp_rel_freqs = 100 * cps_freqs / float(n_time_steps)')
    pyxcd.w('cp_rel_freqs = np.round(cp_rel_freqs, 2)')
    pyxcd.els()

    pyxcd.w('print(\'%-25s\' % \'Unique CPs:\', [\'%5d\' % int(_) for _ in uni_cps])')
    pyxcd.w('print(\'%-25s\' % \'Relative Frequencies (%):\', [\'%5.2f\' % float(_) for _ in cp_rel_freqs])')
    pyxcd.els()

    pyxcd.w(r"print('\nbest_cp_rules_idx_ctr:\n', best_cp_rules_idx_ctr.T)")
    pyxcd.ded(lev=2)

    pyxcd.w('if (curr_n_iter >= temp_adj_iters) and (not temp_adjed) and (run_type == 2):')
    pyxcd.ind()
    pyxcd.w(r'print("\n\n#########Checking for acceptance rate#########")')
    pyxcd.w('print(\'anneal_temp_ini:\', anneal_temp_ini)')

    pyxcd.w('if min_acc_rate <= acc_rate <= max_acc_rate:')
    pyxcd.ind()
    pyxcd.w('print(\'acc_rate (%f%%) is acceptable!\' % acc_rate)')
    pyxcd.w('temp_adjed = 1')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('if ants[0] and ants[1]:')
    pyxcd.ind()
    pyxcd.w('#print(ants)')
    
    pyxcd.w('if acc_rate < min_acc_rate:')
    pyxcd.ind()
    pyxcd.w('print(\'accp_rate (%0.2f%%) is too low!\' % acc_rate)')
    pyxcd.w('ants[0] = [acc_rate, anneal_temp_ini]')
    pyxcd.ded()
    
    pyxcd.w('elif acc_rate > max_acc_rate:')
    pyxcd.ind()
    pyxcd.w('print(\'accp_rate (%0.2f%%) is too high!\' % acc_rate)')
    pyxcd.w('ants[1] = [acc_rate, anneal_temp_ini]')
    pyxcd.ded()
    
    pyxcd.w('#print(anneal_temp_ini)')
    pyxcd.w('anneal_temp_ini = 0.5 * ((ants[1][1] + ants[0][1]))')
    
    pyxcd.w('curr_anneal_temp = anneal_temp_ini')
    pyxcd.w('#print(anneal_temp_ini)')
    pyxcd.w('#print(ants)')
    pyxcd.ded()
    
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('if acc_rate < min_acc_rate:')
    pyxcd.ind()
    pyxcd.w('ants[0] = [acc_rate, anneal_temp_ini]')
    
    pyxcd.w('print(\'accp_rate (%0.2f%%) is too low!\' % acc_rate)')
    pyxcd.w('temp_inc = (1 + ((min_acc_rate) * 0.01))')
    pyxcd.w('print(\'Increasing anneal_temp_ini by %0.2f%%...\' % (100 * (temp_inc - 1)))')
    pyxcd.w('anneal_temp_ini = anneal_temp_ini * temp_inc')
    pyxcd.w('curr_anneal_temp = anneal_temp_ini')
    pyxcd.ded()
    
    pyxcd.w('elif acc_rate > max_acc_rate:')
    pyxcd.ind()
    pyxcd.w('ants[1] = [acc_rate, anneal_temp_ini]')
    
    pyxcd.w('print(\'accp_rate (%0.2f%%) is too high!\' % acc_rate)')
    pyxcd.w('temp_inc = max(1e-6, (1 - ((acc_rate) * 0.01)))')
    pyxcd.w('print(\'Reducing anneal_temp_ini to %0.2f%%...\' %  (100 * (1 - temp_inc)))')
    pyxcd.w('anneal_temp_ini = anneal_temp_ini * temp_inc')
    pyxcd.w('curr_anneal_temp = anneal_temp_ini')
    pyxcd.ded(lev=2)

    pyxcd.w('if curr_temp_adj_iter < max_temp_adj_atmps:')
    pyxcd.ind()
    pyxcd.w('run_type = 1')
    pyxcd.w('curr_n_iter = 0')
    pyxcd.w('curr_m_iter = 0')
    pyxcd.w('best_obj_val = -np.inf')
#     pyxcd.w('pre_obj_val = best_obj_val')
#
#     pyxcd.w('best_accept_iters = 0')
#     pyxcd.w('accept_iters = 0')
#     pyxcd.w('rand_acc_iters = 0')
#     pyxcd.w('reject_iters = 0')
#
#     pyxcd.w('new_iters_ct = 0')
#     pyxcd.w('update_iters_ct = 0')
#     pyxcd.w('rollback_iters_ct = 0')
#
#     pyxcd.w('rand_k = n_cps - 1')
#     pyxcd.w('rand_i = n_pts - 1')
#     pyxcd.w('rand_v = n_fuzz_nos')
#     pyxcd.w('old_v_i_k = n_fuzz_nos')
#
#     pyxcd.w('curr_iters_wo_chng = 0')
#
#     pyxcd.w('curr_temp_adj_iter += 1')
#
#     pyxcd.w('gen_cp_rules(')
#     pyxcd.ind()
#     pyxcd.w('cp_rules,')
#     pyxcd.w('cp_rules_idx_ctr,')
#     pyxcd.w('max_idxs_ct,')
#     pyxcd.w('n_cps,')
#     pyxcd.w('n_pts,')
#     pyxcd.w('n_fuzz_nos,')
#     pyxcd.w('n_cpus)')
#     pyxcd.ded()

    pyxcd.w('continue')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('print(\'#######Could not converge to an acceptable annealing temperature in %d tries!#########\')')
    pyxcd.w('print(\'Terminating optimization....\')')
    pyxcd.w('raise Exception')
    pyxcd.ded(lev=3)

    pyxcd.w('curr_obj_vals_list.append(curr_obj_val)')
    pyxcd.w('best_obj_vals_list.append(best_obj_val)')
    pyxcd.w('acc_rate_list.append(acc_rate)')
    pyxcd.w('cp_pcntge_list.append(ppt_cp_n_vals_arr.copy() / n_time_steps)')
    pyxcd.w('curr_n_iters_list.append(curr_n_iter)')
        
    pyxcd.w('curr_m_iter += 1')
    pyxcd.w('curr_n_iter += 1')
    pyxcd.ded()

#     pyxcd.w('for j in range(n_cps):')
#     pyxcd.ind()
#     pyxcd.w('print(j, loc_mod_ctr[j, :], loc_mod_ctr[j, :].sum(), loc_mod_ctr[j, :].std())')
#     pyxcd.ded()

    pyxcd.w('out_dict = {}')

    pyxcd.w('for key in args_dict:')
    pyxcd.ind()
    pyxcd.w('out_dict[key] = args_dict[key]')
    pyxcd.ded()

    pyxcd.w('out_dict[\'n_pts_calib\'] = n_pts')
    pyxcd.w('out_dict[\'n_fuzz_nos\'] = n_fuzz_nos')
    pyxcd.w('out_dict[\'n_max\'] = n_max')

    if obj_1_flag or obj_3_flag:
        pyxcd.w('out_dict[\'n_stns_calib\'] = n_stns')

    if obj_2_flag or obj_5_flag:
        pyxcd.w('out_dict[\'n_cats_calib\'] = n_cats')

    pyxcd.w('out_dict[\'n_time_steps_calib\'] = n_time_steps')
    pyxcd.w('out_dict[\'last_n_iter\'] = curr_n_iter')
    pyxcd.w('out_dict[\'last_m_iter\'] = curr_m_iter')

    pyxcd.w('out_dict[\'new_iters_ct\'] = new_iters_ct')
    pyxcd.w('out_dict[\'update_iters_ct\'] = update_iters_ct')
    pyxcd.w('out_dict[\'rollback_iters_ct\'] = rollback_iters_ct')

    pyxcd.w('out_dict[\'best_accept_iters\'] = best_accept_iters')
    pyxcd.w('out_dict[\'accept_iters\'] = accept_iters')
    pyxcd.w('out_dict[\'rand_acc_iters\'] = rand_acc_iters')
    pyxcd.w('out_dict[\'reject_iters\'] = reject_iters')
    pyxcd.w('out_dict[\'last_best_accept_n_iter\'] = last_best_accept_n_iter')

    pyxcd.w('out_dict[\'last_obj_val\'] = curr_obj_val')
    pyxcd.w('out_dict[\'best_obj_val\'] = best_obj_val')
    pyxcd.w('out_dict[\'pre_obj_val\'] = pre_obj_val')
    pyxcd.w('out_dict[\'last_anneal_temp\'] = curr_anneal_temp')

    pyxcd.w('out_dict[\'mu_i_k_arr_calib\'] = mu_i_k_arr')

    pyxcd.w('out_dict[\'dofs_arr_calib\'] = dofs_arr')
    pyxcd.w('out_dict[\'best_dofs_arr\'] = best_dofs_arr')

    pyxcd.w('out_dict[\'last_cp_rules\'] = cp_rules')
    pyxcd.w('out_dict[\'best_cp_rules\'] = best_cps')
    pyxcd.w('out_dict[\'best_sel_cps\'] = best_sel_cps')
    pyxcd.w('out_dict[\'last_sel_cps\'] = sel_cps')
    pyxcd.w('out_dict[\'old_sel_cps\'] = old_sel_cps')
    pyxcd.w('out_dict[\'last_cp_rules_idx_ctr\'] = cp_rules_idx_ctr')
    pyxcd.w('out_dict[\'best_cp_rules_idx_ctr\'] = best_cp_rules_idx_ctr')

    pyxcd.w('out_dict[\'curr_obj_vals_arr\'] = np.array(curr_obj_vals_list)')
    pyxcd.w('out_dict[\'best_obj_vals_arr\'] = np.array(best_obj_vals_list)')
    pyxcd.w('out_dict[\'cp_pcntge_arr\'] = np.array(cp_pcntge_list)')
    pyxcd.w('out_dict[\'curr_n_iters_arr\'] = np.array(curr_n_iters_list, '
            'dtype=np.uint64)')
    pyxcd.w('out_dict[\'acc_rate_arr\'] = np.array(acc_rate_list)')

    pyxcd.w('return out_dict')
    pyxcd.ded()

    #==========================================================================
    # write the pyxbld
    #==========================================================================

    write_pyxbld(pyxbldcd)

    #==========================================================================
    # save as pyx, pxd, pyxbld
    #==========================================================================
#     assert pyxcd.level == 0, \
#         'Level should be zero instead of %d' % pyxcd.level
#     assert pxdcd.level == 0, \
#         'Level should be zero instead of %d' % pxdcd.level
    assert pyxbldcd.level == 0, \
        'Level should be zero instead of %d' % pyxbldcd.level

    out_path = os.path.join(out_dir, module_name)
    pyxcd.stf(out_path + '.pyx')
#     pxdcd.stf(out_path + '.pxd')
    pyxbldcd.stf(out_path + '.pyxbld')

#     #==========================================================================
#     # Check for syntax errors
#     #==========================================================================
#     abs_path = os.path.abspath(out_path + '.pyx')
#     arg = (cython, "%s -a" % abs_path)
#     subprocess.call([arg])

    return


if __name__ == '__main__':
    # some text to check git

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(os.getcwd())

    tab = '    '
    nonecheck = False
    boundscheck = False
    wraparound = False
    cdivision = True
    language_level = 3
    infer_types = None
    out_dir = os.getcwd()

    obj_1_flag = True
    obj_2_flag = True
    obj_3_flag = True
    obj_4_flag = True
    obj_5_flag = True

#     obj_1_flag = False
#     obj_2_flag = False
#     obj_3_flag = False
#     obj_4_flag = False
#     obj_5_flag = False

    os.chdir(main_dir)

    assert any([obj_1_flag, obj_2_flag, obj_3_flag, obj_4_flag, obj_5_flag])

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    params_dict['obj_1_flag'] = obj_1_flag
    params_dict['obj_2_flag'] = obj_2_flag
    params_dict['obj_3_flag'] = obj_3_flag
    params_dict['obj_4_flag'] = obj_4_flag
    params_dict['obj_5_flag'] = obj_5_flag

    write_cp_classi_main_lines(params_dict)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
