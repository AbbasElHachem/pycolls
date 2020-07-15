'''
@author: Faizan-Uni-Stuttgart
'''

import os
import timeit
import time
from pathlib import Path

from .core import CodeGenr, write_pyxbld


def write_memb_ftns_lines(params_dict):
    module_name = 'memb_ftns'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    infer_types = params_dict['infer_types']
    out_dir = params_dict['out_dir']
    op_mp_flag = params_dict['op_mp_memb_flag']

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

    pyxcd.w('### op_mp_memb_flag:' + str(op_mp_flag))

    #==========================================================================
    # add imports
    #==========================================================================
    pyxcd.w('import numpy as np')
    pyxcd.w('cimport numpy as np')
    if op_mp_flag:
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

    pyxcd.w('cdef extern from "./fuzzy_ftns.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_D get_tri_mu(const DT_D *g,')
    pyxcd.w('const DT_D *a,')
    pyxcd.w('const DT_D *b,')
    pyxcd.w('const DT_D *c)')
    pyxcd.ded(lev=2)

    #==========================================================================
    # Functions
    #==========================================================================
    pyxcd.w('cdef void calc_membs_dof_cps(')
    pyxcd.ind()
    pyxcd.w('const DT_UL_NP_t[:, :] cp_rules,')
    pyxcd.w('DT_D_NP_t[:, :, :] mu_i_k_arr,')
    pyxcd.w('DT_D_NP_t[:, :, :] cp_dof_arr,')
    pyxcd.w('const DT_D_NP_t[:, :] slp_anom,')
    pyxcd.w('const DT_D_NP_t[:, :] fuzz_nos_arr,')
    pyxcd.w('DT_D_NP_t[:, :] dofs_arr,')
    pyxcd.w('DT_UL_NP_t[:] sel_cps,')
    pyxcd.w('DT_UL_NP_t[:] old_sel_cps,')
    pyxcd.w('DT_UL_NP_t[:] chnge_steps,')
    pyxcd.w('const DT_UL no_cp_val,')
    pyxcd.w('const DT_D p_l,')
    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w('const DT_UL n_time_steps,')
    pyxcd.w('const DT_UL n_pts,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_fuzz_nos,')
    pyxcd.w(') nogil:')

    pxdcd.w('cdef void calc_membs_dof_cps(')
    pxdcd.ind()
    pxdcd.w('const DT_UL_NP_t[:, :] cp_rules,')
    pxdcd.w('DT_D_NP_t[:, :, :] mu_i_k_arr,')
    pxdcd.w('DT_D_NP_t[:, :, :] cp_dof_arr,')
    pxdcd.w('const DT_D_NP_t[:, :] slp_anom,')
    pxdcd.w('const DT_D_NP_t[:, :] fuzz_nos_arr,')
    pxdcd.w('DT_D_NP_t[:, :] dofs_arr,')
    pxdcd.w('DT_UL_NP_t[:] sel_cps,')
    pxdcd.w('DT_UL_NP_t[:] old_sel_cps,')
    pxdcd.w('DT_UL_NP_t[:] chnge_steps,')
    pxdcd.w('const DT_UL no_cp_val,')
    pxdcd.w('const DT_D p_l,')
    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w('const DT_UL n_time_steps,')
    pxdcd.w('const DT_UL n_pts,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_fuzz_nos,')
    pxdcd.w(') nogil')
    pxdcd.ded()

    pyxcd.w("'''")
    pyxcd.w('Calculate memberships, DOFs, selected CPs and time steps with changed')
    pyxcd.w('CPs.')
    pyxcd.w("'''")

    pyxcd.els()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('Py_ssize_t i, j, k, l')
    pyxcd.w('DT_UL curr_fuzz_idx, curr_idxs_sum, best_dof_cp')
    pyxcd.w('DT_D curr_dof, curr_mus_sum, max_dof')
    pyxcd.ded()

    pyxcd.w('# Fill the membership value matrix at each time step,')
    pyxcd.w('# each CP and each point.')
    pyxcd.w('# Select the CP with the greatest DOF for a given step.')
    pyxcd.w('# Set everything to the previous step in case of a roll back.')

    if op_mp_flag:
        pyxcd.w('for i in prange(n_time_steps, schedule=\'dynamic\', '
                'nogil=True, num_threads=n_cpus):')
    else:
        pyxcd.w('for i in range(n_time_steps):')

    pyxcd.ind()
    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('for k in range(n_pts):')
    pyxcd.ind()
    pyxcd.w('curr_fuzz_idx = cp_rules[j, k]')

    pyxcd.w('if curr_fuzz_idx == n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('mu_i_k_arr[i, j, k] = 1.0')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('mu_i_k_arr[i, j, k] = \\')
    pyxcd.ind()
    pyxcd.w('get_tri_mu(&slp_anom[i, k],')
    pyxcd.ind()
    pyxcd.w('&fuzz_nos_arr[curr_fuzz_idx, 0],')
    pyxcd.w('&fuzz_nos_arr[curr_fuzz_idx, 1],')
    pyxcd.w('&fuzz_nos_arr[curr_fuzz_idx, 2])')
    pyxcd.ded(lev=5)

    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('curr_dof = 1.0')

    pyxcd.w('for l in range(n_fuzz_nos):')
    pyxcd.ind()
    pyxcd.w('curr_idxs_sum = 0')
    pyxcd.w('curr_mus_sum = 0')

    pyxcd.w('for k in range(n_pts):')
    pyxcd.ind()
    pyxcd.w('if cp_rules[j, k] != l:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()

    pyxcd.w('curr_idxs_sum = curr_idxs_sum + 1')
    pyxcd.w('curr_mus_sum = curr_mus_sum + (mu_i_k_arr[i, j, k]**p_l)')
    pyxcd.ded()

    pyxcd.w('if curr_idxs_sum:')
    pyxcd.ind()
    pyxcd.w('curr_mus_sum = (curr_mus_sum / curr_idxs_sum)**(1.0 / p_l)')
    pyxcd.ded()

    pyxcd.w('cp_dof_arr[i, j, l] = curr_mus_sum')

    pyxcd.w('curr_dof = curr_mus_sum * curr_dof')
    pyxcd.ded()

    pyxcd.w('dofs_arr[i, j] = curr_dof')
    pyxcd.ded()

    pyxcd.w('best_dof_cp = no_cp_val')
    pyxcd.w('max_dof = 1e-5')
    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('if dofs_arr[i, j] > max_dof:')
    pyxcd.ind()
    pyxcd.w('max_dof = dofs_arr[i, j]')
    pyxcd.w('best_dof_cp = j')
    pyxcd.ded(lev=2)

    pyxcd.w('old_sel_cps[i] = no_cp_val')
    pyxcd.w('sel_cps[i] = best_dof_cp')

    pyxcd.w('if sel_cps[i] != old_sel_cps[i]:')
    pyxcd.ind()
    pyxcd.w('chnge_steps[i] = 1')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('chnge_steps[i] = 0')
    pyxcd.ded(lev=2)

    pyxcd.w('return')
    pyxcd.ded()

    # the update function
    pyxcd.w('cdef void update_membs_dof_cps(')
    pyxcd.ind()
    pyxcd.w('const DT_UL pre_fuzz_idx,')
    pyxcd.w('const DT_UL curr_fuzz_idx,')
    pyxcd.w('const DT_UL curr_cp,')
    pyxcd.w('const DT_UL curr_pt,')
    pyxcd.w('const DT_UL_NP_t[:, :] cp_rules,')
    pyxcd.w('DT_D_NP_t[:, :, :] mu_i_k_arr,')
    pyxcd.w('DT_D_NP_t[:, :, :] cp_dof_arr,')
    pyxcd.w('const DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pyxcd.w('const DT_D_NP_t[:, :] slp_anom,')
    pyxcd.w('const DT_D_NP_t[:, :] fuzz_nos_arr,')
    pyxcd.w('DT_D_NP_t[:, :] dofs_arr,')
    pyxcd.w('DT_UL_NP_t[:] sel_cps,')
    pyxcd.w('DT_UL_NP_t[:] old_sel_cps,')
    pyxcd.w('DT_UL_NP_t[:] chnge_steps,')
    pyxcd.w('const DT_UL no_cp_val,')
    pyxcd.w('const DT_D p_l,')
    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w('const DT_UL n_time_steps,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_fuzz_nos,')
    pyxcd.w(') nogil:')

    pxdcd.w('cdef void update_membs_dof_cps(')
    pxdcd.ind()
    pxdcd.w('const DT_UL pre_fuzz_idx,')
    pxdcd.w('const DT_UL curr_fuzz_idx,')
    pxdcd.w('const DT_UL curr_cp,')
    pxdcd.w('const DT_UL curr_pt,')
    pxdcd.w('const DT_UL_NP_t[:, :] cp_rules,')
    pxdcd.w('DT_D_NP_t[:, :, :] mu_i_k_arr,')
    pxdcd.w('DT_D_NP_t[:, :, :] cp_dof_arr,')
    pxdcd.w('const DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pxdcd.w('const DT_D_NP_t[:, :] slp_anom,')
    pxdcd.w('const DT_D_NP_t[:, :] fuzz_nos_arr,')
    pxdcd.w('DT_D_NP_t[:, :] dofs_arr,')
    pxdcd.w('DT_UL_NP_t[:] sel_cps,')
    pxdcd.w('DT_UL_NP_t[:] old_sel_cps,')
    pxdcd.w('DT_UL_NP_t[:] chnge_steps,')
    pxdcd.w('const DT_UL no_cp_val,')
    pxdcd.w('const DT_D p_l,')
    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w('const DT_UL n_time_steps,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_fuzz_nos,')
    pxdcd.w(') nogil')
    pxdcd.ded()

    pyxcd.w("'''")
    pyxcd.w('Update memberships and DOFs with changed CPs.')
    pyxcd.w("'''")

    pyxcd.els()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('Py_ssize_t i, j, l')
    pyxcd.w('int pre_cond, curr_cond')
    pyxcd.w('DT_UL pre_idxs_sum, best_dof_cp, curr_idxs_sum')
    pyxcd.w('DT_D curr_dof, curr_mus_sum, max_dof, pre_mus_sum')
    pyxcd.w('DT_D f1, f2, f3')
    pyxcd.ded()

    pyxcd.w('if curr_fuzz_idx != n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('f1 = fuzz_nos_arr[curr_fuzz_idx, 0]')
    pyxcd.w('f2 = fuzz_nos_arr[curr_fuzz_idx, 1]')
    pyxcd.w('f3 = fuzz_nos_arr[curr_fuzz_idx, 2]')
    pyxcd.w('curr_idxs_sum = cp_rules_idx_ctr[curr_cp, curr_fuzz_idx]')
    pyxcd.w('curr_cond = 1')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('curr_cond = 0')
    pyxcd.ded()

    pyxcd.w('if pre_fuzz_idx != n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('pre_cond = 1')
    pyxcd.w('pre_idxs_sum = cp_rules_idx_ctr[curr_cp, pre_fuzz_idx]')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('pre_cond = 0')
    pyxcd.ded()

    if op_mp_flag:
        pyxcd.w('for i in prange(n_time_steps, schedule=\'static\', '
                'nogil=True, num_threads=n_cpus):')
    else:
        pyxcd.w('for i in range(n_time_steps):')

    pyxcd.ind()
    pyxcd.w('# remove old')
    pyxcd.w('if pre_cond:')
    pyxcd.ind()
    pyxcd.w('if pre_idxs_sum > 0:')
    pyxcd.ind()
    pyxcd.w(
        'pre_mus_sum = (cp_dof_arr[i, curr_cp, pre_fuzz_idx]**p_l) * (pre_idxs_sum + 1)')

    pyxcd.w(
        'pre_mus_sum = pre_mus_sum - (mu_i_k_arr[i, curr_cp, curr_pt]**p_l)')

    pyxcd.els()
    pyxcd.w('if (pre_mus_sum < 0):  # numerical errors around -1e-18')
    pyxcd.ind()
    pyxcd.w('pre_mus_sum = 0.0')
    pyxcd.ded()
    pyxcd.w(
        'cp_dof_arr[i, curr_cp, pre_fuzz_idx] = (pre_mus_sum / pre_idxs_sum)**(1.0 / p_l)')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('cp_dof_arr[i, curr_cp, pre_fuzz_idx] = 0.0')
    pyxcd.ded(lev=2)
    pyxcd.w('# update the mu value at the point')
    pyxcd.w('if curr_fuzz_idx == n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('mu_i_k_arr[i, curr_cp, curr_pt] = 1.0')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('mu_i_k_arr[i, curr_cp, curr_pt] = \\')
    pyxcd.ind()
    pyxcd.w('get_tri_mu(&slp_anom[i, curr_pt], &f1, &f2, &f3)')
    pyxcd.ded(lev=2)
    pyxcd.w('# add new')
    pyxcd.w('if curr_cond:')
    pyxcd.ind()
    pyxcd.w(
        'curr_mus_sum = (cp_dof_arr[i, curr_cp, curr_fuzz_idx]**p_l) * (curr_idxs_sum - 1)')
    pyxcd.w(
        'curr_mus_sum = curr_mus_sum + (mu_i_k_arr[i, curr_cp, curr_pt]**p_l)')
    pyxcd.w(
        'cp_dof_arr[i, curr_cp, curr_fuzz_idx] = (curr_mus_sum / curr_idxs_sum)**(1.0 / p_l)')
    pyxcd.ded()
    pyxcd.w('# update new dofs for pre and curr cps')
    pyxcd.w('curr_dof = 1.0')
    pyxcd.w('for l in range(n_fuzz_nos):')
    pyxcd.ind()
    pyxcd.w('curr_mus_sum = cp_dof_arr[i, curr_cp, l]')
    pyxcd.w('curr_dof = curr_mus_sum * curr_dof')
    pyxcd.ded()
    pyxcd.w('dofs_arr[i, curr_cp] = curr_dof')

    pyxcd.w('best_dof_cp = no_cp_val')
    pyxcd.w('max_dof = 1e-5')
    pyxcd.w('for j in range(n_cps):')
    pyxcd.ind()
    pyxcd.w('if dofs_arr[i, j] > max_dof:')
    pyxcd.ind()
    pyxcd.w('max_dof = dofs_arr[i, j]')
    pyxcd.w('best_dof_cp = j')
    pyxcd.ded(lev=2)
    pyxcd.w('old_sel_cps[i] = sel_cps[i]')
    pyxcd.w('sel_cps[i] = best_dof_cp')

    pyxcd.w('if sel_cps[i] != old_sel_cps[i]:')
    pyxcd.ind()
    pyxcd.w('chnge_steps[i] = 1')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('chnge_steps[i] = 0')
    pyxcd.ded(lev=2)
    pyxcd.w('return')
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

    os.chdir(main_dir)

    params_dict = {}
    params_dict['tab'] = tab
    params_dict['nonecheck'] = nonecheck
    params_dict['boundscheck'] = boundscheck
    params_dict['wraparound'] = wraparound
    params_dict['cdivision'] = cdivision
    params_dict['language_level'] = language_level
    params_dict['infer_types'] = infer_types
    params_dict['out_dir'] = out_dir

    write_memb_ftns_lines(params_dict)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
