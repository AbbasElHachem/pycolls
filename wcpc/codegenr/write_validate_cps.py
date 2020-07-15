'''
@author: Faizan-Uni-Stuttgart

Template
'''

import os
import timeit
import time
from pathlib import Path

from .core import CodeGenr, write_pyxbld


def write_validate_cps_lines(params_dict):
    module_name = 'validate_cps'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    out_dir = params_dict['out_dir']

    pyxcd = CodeGenr(tab=tab)
    pxdcd = CodeGenr(tab=tab)
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

    pxdcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pxdcd.w('# cython: boundscheck=%s' % boundscheck)
    pxdcd.w('# cython: wraparound=%s' % str(wraparound))
    pxdcd.w('# cython: cdivision=%s' % str(cdivision))
    pxdcd.w('# cython: language_level=%d' % int(language_level))
    pxdcd.els()

    _ = ';'.join(map(str, [obj_1_flag,
                           obj_2_flag,
                           obj_3_flag,
                           obj_4_flag,
                           obj_5_flag]))
    pyxcd.w('### obj_ftns:' + _)
    pyxcd.els()

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    pyxcd.els()
    pyxcd.w('from .validate_cps_qual cimport validate_cps_quality')
    pyxcd.w('from .memb_ftns cimport calc_membs_dof_cps')
    pyxcd.els()

    pxdcd.w('import numpy as np')
    pxdcd.w('cimport numpy as np')
    pxdcd.els()

    #==========================================================================
    # declare types
    #==========================================================================
    pxdcd.w('ctypedef double DT_D')
    pxdcd.w('ctypedef unsigned long DT_UL')
    pxdcd.w('ctypedef long long DT_LL')
    pxdcd.w('ctypedef unsigned long long DT_ULL')
    pxdcd.w('ctypedef np.float64_t DT_D_NP_t')
    pxdcd.w('ctypedef np.uint64_t DT_UL_NP_t')
    pxdcd.els()

    pyxcd.w('DT_D_NP = np.float64')
    pyxcd.w('DT_UL_NP = np.uint64')
    pyxcd.els()

    pxdcd.w('DT_D_NP = np.float64')
    pxdcd.w('DT_UL_NP = np.uint64')
    pxdcd.els(2)

    #==========================================================================
    # add external imports
    #==========================================================================

    pyxcd.w('cdef extern from "math.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('bint isnan(DT_D x)')
    pyxcd.ded(lev=2)
    pyxcd.els()

    #==========================================================================
    # Functions
    #==========================================================================

    pyxcd.w('cpdef validate_cps(dict args_dict):')
    pxdcd.w('cpdef validate_cps(dict args_dict)')

    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('# ulongs')
    pyxcd.w('DT_UL n_cps, n_pts, n_time_steps, n_fuzz_nos')
    pyxcd.w('DT_UL no_cp_val, run_type, n_stns, n_cpus, msgs')
    pyxcd.els()

    pyxcd.w('# doubles')
    pyxcd.w('#DT_D curr_obj_val')
    pyxcd.els()

    pyxcd.w('# doubles for obj. ftns.')
    pyxcd.w('DT_D p_l, min_abs_ppt_thresh')

    pyxcd.w('DT_D ppt_lo_thresh')
    pyxcd.w('DT_D ppt_hi_thresh')

    pyxcd.els()

    pyxcd.w('# 1D ulong arrays')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=1, mode=\'c\'] chnge_steps')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=1, mode=\'c\'] sel_cps, old_sel_cps')
    pyxcd.els()

    pyxcd.w('# 2D ulong arrays')
    pyxcd.w('np.ndarray[DT_UL_NP_t, ndim=2, mode=\'c\'] cp_rules')
    pyxcd.els()

    pyxcd.w('# 2D double arrays')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] slp_anom, fuzz_nos_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] dofs_arr')
    pyxcd.els()

    pyxcd.w('# 3D double arrays')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] mu_i_k_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=3, mode=\'c\'] cp_dof_arr')
    pyxcd.els()

    pyxcd.w('# arrays for all obj. ftns.')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] ppt_pi_lo_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] ppt_pi_hi_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=1, mode=\'c\'] ppt_mean_arr')

    pyxcd.els()

    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] in_ppt_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] wet_idxs_arr')
    pyxcd.w('np.ndarray[DT_D_NP_t, ndim=2, mode=\'c\'] other_qual_prms_arr')
    pyxcd.ded()

    pyxcd.w('# a dict to hold all returned variables')
    pyxcd.w('out_dict = {}')
    pyxcd.w('for key in args_dict:')
    pyxcd.ind()
    pyxcd.w('out_dict[key] = args_dict[key]')
    pyxcd.ded()

    pyxcd.w('# read everything from the given dict. Must do explicitly!')

    pyxcd.w('ppt_lo_thresh = args_dict[\'ppt_lo_thresh\']')
    pyxcd.w('ppt_hi_thresh = args_dict[\'ppt_hi_thresh\'] ')

    pyxcd.w('min_abs_ppt_thresh = args_dict[\'min_abs_ppt_thresh\']')
    pyxcd.w('no_cp_val = args_dict[\'no_cp_val\']')
    pyxcd.w('p_l = args_dict[\'p_l\']')
    pyxcd.w('fuzz_nos_arr = args_dict[\'fuzz_nos_arr\']')
    pyxcd.w('cp_rules = args_dict[\'best_cp_rules\']')
    pyxcd.w('n_cpus = args_dict[\'n_cpus\']')
    pyxcd.els()

    pyxcd.w('n_fuzz_nos = fuzz_nos_arr.shape[0]')
    pyxcd.w('n_cps = cp_rules.shape[0]')
    pyxcd.w('assert n_cps >= 2, \'n_cps cannot be less than 2!\'')
    pyxcd.els()

    pyxcd.w('if \'msgs\' in args_dict:')
    pyxcd.ind()
    pyxcd.w('msgs = <DT_UL> args_dict[\'msgs\']')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('msgs = 0')
    pyxcd.ded()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\n')")

    pyxcd.w('print(\'ppt_lo_thresh:\', ppt_lo_thresh)')
    pyxcd.w('print(\'ppt_hi_thresh:\', ppt_hi_thresh)')

    pyxcd.w('print(\'min_abs_ppt_thresh:\', min_abs_ppt_thresh)')
    pyxcd.w('print(\'n_cps:\', n_cps)')
    pyxcd.w('print(\'n_cpus:\', n_cpus)')
    pyxcd.w('print(\'no_cp_val:\', no_cp_val)')
    pyxcd.w('print(\'p_l:\', p_l)')
    pyxcd.w(r"print('fuzz_nos_arr:\n', fuzz_nos_arr)")
    pyxcd.w(r"print('\n')")
    pyxcd.ded()

    pyxcd.w('calib_valid_list = [\'_calib\', \'_valid\']  # dont change')
    pyxcd.w('run_type = 1  # dont change')
    pyxcd.els()

    pyxcd.w('for lab in calib_valid_list:')
    pyxcd.ind()
    pyxcd.w('slp_anom = args_dict[\'slp_anom\' + lab]')
    pyxcd.w('in_ppt_arr = args_dict[\'in_ppt_arr\' + lab]')
    pyxcd.els()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\n\n\nRun type:', lab[1:].upper())")
    pyxcd.w(r"print('\nslp_anom shape: (%d, %d)' % "
            r"(slp_anom.shape[0], slp_anom.shape[1]))")
    pyxcd.w(r"print('\nin_ppt_arr shape: (%d, %d)\n' % "
            r"(in_ppt_arr.shape[0], in_ppt_arr.shape[1]))")
    pyxcd.ded()

    pyxcd.w('# initialize the required variables')
    pyxcd.w('n_pts = slp_anom.shape[1]')
    pyxcd.w('n_stns = in_ppt_arr.shape[1]')
    pyxcd.w('n_time_steps = slp_anom.shape[0]')
    pyxcd.els()

    pyxcd.w('mu_i_k_arr = np.zeros(shape=(n_time_steps, n_cps, n_pts), '
            'dtype=DT_D_NP)')
    pyxcd.w('cp_dof_arr = np.zeros(shape=(n_time_steps, n_cps, n_fuzz_nos), '
            'dtype=DT_D_NP)')
    pyxcd.els()
    pyxcd.w('sel_cps = np.zeros(n_time_steps, dtype=DT_UL_NP)')
    pyxcd.w('old_sel_cps = sel_cps.copy()')
    pyxcd.els()

    pyxcd.w('chnge_steps = np.zeros(n_time_steps, dtype=DT_UL_NP)')
    pyxcd.w('dofs_arr = np.zeros((n_time_steps, n_cps), dtype=DT_D_NP)')
    pyxcd.els()

    pyxcd.w('# initialize the obj. ftn. variables')
    pyxcd.w('wet_idxs_arr = np.zeros((n_cps, n_stns), dtype=DT_D_NP)')
    pyxcd.w('other_qual_prms_arr = np.zeros((4, n_stns), dtype=DT_D_NP)')
    pyxcd.els()

    pyxcd.w('ppt_pi_lo_arr = np.zeros(n_stns, dtype=DT_D_NP)')
    pyxcd.w('ppt_pi_hi_arr = np.zeros(n_stns, dtype=DT_D_NP)')
    pyxcd.w('ppt_mean_arr = np.zeros(n_stns, dtype=DT_D_NP)')

    pyxcd.els()

    pyxcd.w('# fill some arrays used for obj. ftns.')
    pyxcd.w('for m in range(n_stns):')
    pyxcd.ind()

    pyxcd.w(
        'ppt_pi_lo_arr[m] = np.sum(in_ppt_arr[:, m] > ppt_lo_thresh) / n_time_steps')
    pyxcd.w('assert ((not isnan(ppt_pi_lo_arr[m])) and (ppt_pi_lo_arr[m]> 0))')
    pyxcd.els()

    pyxcd.w(
        'ppt_pi_hi_arr[m] = np.sum(in_ppt_arr[:, m] > ppt_hi_thresh) / n_time_steps')
    pyxcd.w(
        'assert ((not isnan(ppt_pi_hi_arr[m])) and (ppt_pi_hi_arr[m] > 0))')
    pyxcd.els()

    pyxcd.w('ppt_mean_arr[m] = np.mean(in_ppt_arr[:, m])')
    pyxcd.w('assert ((not isnan(ppt_mean_arr[m])) and (ppt_mean_arr[m]> 0))')

    pyxcd.ded()

    pyxcd.w('# fill/update the membership, DOF and selected CPs arrays')
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
    pyxcd.ded()

    pyxcd.w('validate_cps_quality(')
    pyxcd.ind()
    pyxcd.w('in_ppt_arr,')
    pyxcd.w('wet_idxs_arr,')
    pyxcd.w('other_qual_prms_arr,')
    pyxcd.w('ppt_pi_lo_arr,')
    pyxcd.w('ppt_pi_hi_arr,')
    pyxcd.w('ppt_mean_arr,')
    pyxcd.w('sel_cps,')
    pyxcd.w('min_abs_ppt_thresh,')
    pyxcd.w('ppt_lo_thresh,')
    pyxcd.w('ppt_hi_thresh,')
    pyxcd.w('n_cpus,')
    pyxcd.w('n_stns,')
    pyxcd.w('n_cps,')
    pyxcd.w('n_time_steps)')
    pyxcd.ded()

    pyxcd.w('out_dict[\'ppt_pi_lo_arr\' + lab] = ppt_pi_lo_arr')
    pyxcd.w('out_dict[\'ppt_pi_hi_arr\' + lab] = ppt_pi_hi_arr')
    pyxcd.w('out_dict[\'ppt_mean_arr\' + lab] = ppt_mean_arr')

    pyxcd.els()

    pyxcd.w('out_dict[\'n_pts\' + lab] = n_pts')
    pyxcd.w('out_dict[\'n_time_steps\' + lab] = n_time_steps')
    pyxcd.w('out_dict[\'mu_i_k_arr\' + lab] = mu_i_k_arr')
    pyxcd.w('out_dict[\'cp_dof_arr\' + lab] = cp_dof_arr')
    pyxcd.w('out_dict[\'dofs_arr\' + lab] = dofs_arr')
    pyxcd.w('out_dict[\'wet_idxs_arr\' + lab] =  wet_idxs_arr')
    pyxcd.w('out_dict[\'other_qual_prms_arr\' + lab] = other_qual_prms_arr')
    pyxcd.w('out_dict[\'sel_cps\' + lab] = sel_cps ')
    pyxcd.ded()

    pyxcd.w('out_dict[\'n_fuzz_nos\'] = n_fuzz_nos')
    pyxcd.els()

    pyxcd.w('if msgs:')
    pyxcd.ind()
    pyxcd.w(r"print('\n\n\n')")
    pyxcd.ded()

    pyxcd.w('return out_dict')
    pyxcd.ded()

    #==========================================================================
    # write the pyxbld
    #==========================================================================

    write_pyxbld(pyxbldcd)

    #==========================================================================
    # save as pyx, pxd, pyxbld
    #==========================================================================
    assert pyxcd.level == 0, \
        'Level should be zero instead of %d' % pyxcd.level
    assert pxdcd.level == 0, \
        'Level should be zero instead of %d' % pxdcd.level
    assert pyxbldcd.level == 0, \
        'Level should be zero instead of %d' % pyxbldcd.level

    out_path = os.path.join(out_dir, module_name)
    pyxcd.stf(out_path + '.pyx')
    pxdcd.stf(out_path + '.pxd')
    pyxbldcd.stf(out_path + '.pyxbld')

#     #==========================================================================
#     # Check for syntax errors
#     #==========================================================================
#     abs_path = os.path.abspath(out_path + '.pyx')
#     arg = (cython, "%s -a" % abs_path)
#     subprocess.call([arg])

    return


if __name__ == '__main__':
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

#     nonecheck = True
#     boundscheck = True
#     wraparound = True
#     cdivision = False

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

    write_validate_cps_lines(params_dict)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
