'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

from .core import CodeGenr, write_pyxbld


def write_validate_cps_qual_lines(params_dict):
    module_name = 'validate_cps_qual'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    infer_types = params_dict['infer_types']
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
    pyxcd.w('# cython: infer_types(%s)' % str(infer_types))
    pyxcd.els()

    pxdcd.w('# cython: nonecheck=%s' % str(nonecheck))
    pxdcd.w('# cython: boundscheck=%s' % boundscheck)
    pxdcd.w('# cython: wraparound=%s' % str(wraparound))
    pxdcd.w('# cython: cdivision=%s' % str(cdivision))
    pxdcd.w('# cython: language_level=%d' % int(language_level))
    pxdcd.w('# cython: infer_types(%s)' % str(infer_types))
    pxdcd.els()

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    pyxcd.w('from cython.parallel import prange')
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
    pyxcd.els()
    pyxcd.w('cdef extern from "math.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_D log(DT_D x)')
    pyxcd.w('DT_D abs(DT_D x)')
    pyxcd.ded(lev=2)
    pyxcd.els()

    #==========================================================================
    # Functions
    #==========================================================================
    pyxcd.w('cdef void validate_cps_quality(')
    pyxcd.ind()
    pyxcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
    pyxcd.w('DT_D_NP_t[:, :] wet_idxs_arr,')
    pyxcd.w('DT_D_NP_t[:, :] other_qual_prms_arr,')

    pyxcd.w('const DT_D_NP_t[:] ppt_pi_lo_arr,')
    pyxcd.w('const DT_D_NP_t[:] ppt_pi_hi_arr,')
    pyxcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    pyxcd.w('const DT_UL_NP_t[:] sel_cps,')
    pyxcd.w('const DT_D min_abs_ppt_thresh,')

    pyxcd.w('const DT_D ppt_lo_thresh,')
    pyxcd.w('const DT_D ppt_hi_thresh,')

    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w('const DT_UL n_stns,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_time_steps,')
    pyxcd.w(') nogil:')

    pxdcd.w('cdef void validate_cps_quality(')
    pxdcd.ind()
    pxdcd.w('const DT_D_NP_t[:, :] in_ppt_arr,')
    pxdcd.w('DT_D_NP_t[:, :] wet_idxs_arr,')
    pxdcd.w('DT_D_NP_t[:, :] other_qual_prms_arr,')

    pxdcd.w('const DT_D_NP_t[:] ppt_pi_lo_arr,')
    pxdcd.w('const DT_D_NP_t[:] ppt_pi_hi_arr,')
    pxdcd.w('const DT_D_NP_t[:] ppt_mean_arr,')

    pxdcd.w('const DT_UL_NP_t[:] sel_cps,')
    pxdcd.w('const DT_D min_abs_ppt_thresh,')

    pxdcd.w('const DT_D ppt_lo_thresh,')
    pxdcd.w('const DT_D ppt_hi_thresh,')

    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w('const DT_UL n_stns,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_time_steps,')
    pxdcd.w(') nogil')
    pxdcd.ded()

    pyxcd.els()
    pyxcd.w(r"'''")
    pyxcd.w('Calculate quality paramters for a given classification.')
    pyxcd.els()
    pyxcd.w('wet_idxs_arr has the shape: (no. of cps, no. of stns)')
    pyxcd.w('other_qual_prms_arr has the shape: '
            '(no. qual. params., no. of stns)')
    pyxcd.w(r"'''")
    pyxcd.els()

    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('Py_ssize_t i, j, m')
    pyxcd.els()

    pyxcd.w('DT_D _')
    pyxcd.w('DT_D curr_ppt, curr_n_vals')

    pyxcd.w('DT_D cp_pi_lo, curr_ppt_pi_lo_diff')
    pyxcd.w('DT_D cp_pi_hi, curr_ppt_pi_hi_diff')
    pyxcd.w('DT_D cp_ppt_mean')

    pyxcd.ded()

    pyxcd.w(
        'for m in prange(n_stns, schedule=\'dynamic\', nogil=True, num_threads=n_cpus):')
    pyxcd.ind()

    pyxcd.w('curr_ppt_pi_lo_diff = 0')
    pyxcd.w('curr_ppt_pi_hi_diff = 0')

    pyxcd.els()

    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('curr_n_vals = 0')

    pyxcd.w('cp_pi_lo = 0')
    pyxcd.w('cp_pi_hi = 0')
    pyxcd.w('cp_ppt_mean = 0')

    pyxcd.els()

    pyxcd.w('for i in range(n_time_steps):')
    pyxcd.ind()
    pyxcd.w('if sel_cps[i] != j:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()

    pyxcd.w('curr_ppt = in_ppt_arr[i, m]')
    pyxcd.w('curr_n_vals = curr_n_vals + 1')
    pyxcd.els()

    pyxcd.w('cp_ppt_mean = cp_ppt_mean + curr_ppt')
    pyxcd.els()
    pyxcd.w('if curr_ppt < ppt_lo_thresh:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()
    pyxcd.w('cp_pi_lo = cp_pi_lo + 1')
    pyxcd.els()
    pyxcd.w('if curr_ppt < ppt_hi_thresh:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()
    pyxcd.w('cp_pi_hi = cp_pi_hi + 1')
    pyxcd.els()

    pyxcd.ded()

    pyxcd.w('if curr_n_vals == 0:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()

    pyxcd.w('cp_pi_lo = cp_pi_lo / curr_n_vals')
    pyxcd.w('curr_ppt_pi_lo_diff = curr_ppt_pi_lo_diff + '
            'curr_n_vals * (cp_pi_lo - ppt_pi_lo_arr[m])**2')
    pyxcd.els()
    pyxcd.w('cp_pi_hi = cp_pi_hi / curr_n_vals')
    pyxcd.w('curr_ppt_pi_hi_diff = curr_ppt_pi_hi_diff + '
            'curr_n_vals * (cp_pi_hi - ppt_pi_hi_arr[m])**2')
    pyxcd.els()
    pyxcd.w('cp_ppt_mean = cp_ppt_mean / curr_n_vals')
    pyxcd.w('if ((ppt_mean_arr[m] > min_abs_ppt_thresh) and curr_n_vals):')
    pyxcd.ind()
    pyxcd.w('_ = cp_ppt_mean / ppt_mean_arr[m]')
    pyxcd.w('if _ <= 0:')
    pyxcd.ind()
    pyxcd.w('_ = 1e-100')
    pyxcd.ded()

    pyxcd.w('wet_idxs_arr[j, m] = _')
    pyxcd.els()

    pyxcd.w('other_qual_prms_arr[2, m] = other_qual_prms_arr[2, m] + '
            '((abs(_ - 1)) * (curr_n_vals / n_time_steps))')
    pyxcd.els()

    pyxcd.w('other_qual_prms_arr[3, m] = other_qual_prms_arr[3, m] + '
            '(abs(log(_)) * (curr_n_vals / n_time_steps))')
    pyxcd.ded()

    pyxcd.ded()
    pyxcd.w('other_qual_prms_arr[0, m] = '
            '(curr_ppt_pi_lo_diff / n_time_steps)**0.5')

    pyxcd.w('other_qual_prms_arr[1, m] = '
            '(curr_ppt_pi_hi_diff / n_time_steps)**0.5')

    pyxcd.ded()
    pyxcd.w('return')
    pyxcd.ded(False)

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

# if __name__ == '__main__':
#     print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
#     START = timeit.default_timer()  # to get the runtime of the program
#
#     main_dir = Path(os.getcwd())
#
#     tab = '    '
#     nonecheck = False
#     boundscheck = False
#     wraparound = False
#     cdivision = True
#     language_level = 3
#     infer_types = None
#     out_dir = os.getcwd()
#
#     obj_1_flag = True
#     obj_2_flag = True
#     obj_3_flag = True
#
# #     obj_1_flag = False
# #     obj_2_flag = False
# #     obj_3_flag = False
#
#     os.chdir(main_dir)
#
#     params_dict = {}
#     params_dict['tab'] = tab
#     params_dict['nonecheck'] = nonecheck
#     params_dict['boundscheck'] = boundscheck
#     params_dict['wraparound'] = wraparound
#     params_dict['cdivision'] = cdivision
#     params_dict['language_level'] = language_level
#     params_dict['infer_types'] = infer_types
#     params_dict['out_dir'] = out_dir
#
#     params_dict['obj_1_flag'] = obj_1_flag
#     params_dict['obj_2_flag'] = obj_2_flag
#     params_dict['obj_3_flag'] = obj_3_flag
#
#     write_validate_cps_qual_lines(params_dict)
#
#     STOP = timeit.default_timer()  # Ending time
#     print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
#            ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
