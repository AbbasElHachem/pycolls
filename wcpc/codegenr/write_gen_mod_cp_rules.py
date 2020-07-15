'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

from .core import CodeGenr, write_pyxbld


def write_gen_mod_cp_rules_lines(params_dict):
    module_name = 'gen_mod_cp_rules'

    tab = params_dict['tab']
    nonecheck = params_dict['nonecheck']
    boundscheck = params_dict['boundscheck']
    wraparound = params_dict['wraparound']
    cdivision = params_dict['cdivision']
    language_level = params_dict['language_level']
    infer_types = params_dict['infer_types']
    out_dir = params_dict['out_dir']

    op_mp_flag = any([params_dict['op_mp_memb_flag'],
                      params_dict['op_mp_obj_ftn_flag']])

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
    pyxcd.els()
    pyxcd.w('cdef extern from "stdio.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('int printf(const char *x)')
    pyxcd.ded(lev=2)

    pyxcd.els()
    pyxcd.w('cdef extern from "./rand_gen.h" nogil:')
    pyxcd.ind()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_D rand_c()')
    pyxcd.w('void warm_up()  # call this at least once')
    pyxcd.w('void re_seed(DT_ULL x)  # calls warm_up as well')
    pyxcd.ded(lev=2)
    pyxcd.w('warm_up()')
    pyxcd.els(2)

    pyxcd.w('cdef void gen_cp_rules(')
    pyxcd.ind()
    pyxcd.w('DT_UL_NP_t[:, :] cp_rules,')
    pyxcd.w('DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pyxcd.w('const DT_UL max_idxs_ct,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_pts,')
    pyxcd.w('const DT_UL n_fuzz_nos,')
    pyxcd.w('const DT_UL n_cpus,')
    pyxcd.w(') nogil:')

    pxdcd.w('cdef void gen_cp_rules(')
    pxdcd.ind()
    pxdcd.w('DT_UL_NP_t[:, :] cp_rules,')
    pxdcd.w('DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pxdcd.w('const DT_UL max_idxs_ct,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_pts,')
    pxdcd.w('const DT_UL n_fuzz_nos,')
    pxdcd.w('const DT_UL n_cpus,')
    pxdcd.w(') nogil')
    pxdcd.ded()

    pyxcd.els()
    pyxcd.w(r"'''")
    pyxcd.w('Generate CP rules, given the maximum number of a rule that a')
    pyxcd.w('CP can have (max_idxs_ct).')
    pyxcd.els()

    pyxcd.w('The number of indicies assigned inside each CP for a given rule')
    pyxcd.w('are between zero and max_idxs_ct.')
    pyxcd.w(r"'''")

    pyxcd.els()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('Py_ssize_t j, k, l')
    pyxcd.w('DT_UL curr_idxs_ct, curr_idxs_ctr')
    pyxcd.w('DT_UL rand_i, rand_v')
    pyxcd.w('DT_UL max_iters = 1000000, curr_iter_ctr = 0')
    pyxcd.ded()

    if op_mp_flag:
        pyxcd.w('for j in prange(n_cps, schedule=\'static\', '
                'nogil=True, num_threads=n_cpus):')
    else:
        pyxcd.w('for j in range(n_cps):')

    pyxcd.ind()
    pyxcd.w('for k in range(n_pts):')
    pyxcd.ind()
    pyxcd.w('cp_rules[j, k] = n_fuzz_nos')
    pyxcd.ded()

    pyxcd.w('curr_iter_ctr = 0')
    pyxcd.w('for l in range(n_fuzz_nos):')
    pyxcd.ind()
    pyxcd.w('curr_idxs_ct = <DT_UL> (rand_c() * (max_idxs_ct + 1))')
    pyxcd.w('cp_rules_idx_ctr[j, l] = curr_idxs_ct')

    pyxcd.els()
    pyxcd.w('curr_idxs_ctr = 0')
    pyxcd.w('while (curr_idxs_ctr < curr_idxs_ct):')
    pyxcd.ind()
    pyxcd.w('curr_iter_ctr = curr_iter_ctr + 1')

    pyxcd.w('if curr_iter_ctr > max_iters:')
    pyxcd.ind()
    pyxcd.w(r'printf("\n\n\n\n########Too many iterations in gen_cp_rules!'
            r'########\n\n\n\n")')
    pyxcd.w('break')
    pyxcd.ded()

    pyxcd.w('rand_i = <DT_UL> (rand_c() * n_pts)')
    pyxcd.w('if cp_rules[j, rand_i] != n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded()

    pyxcd.w('cp_rules[j, rand_i] = l')
    pyxcd.w('curr_idxs_ctr = curr_idxs_ctr + 1')
    pyxcd.ded(lev=3)

    pyxcd.w('return')
    pyxcd.ded()

    assert pyxcd.level == 0, \
        'Level should be zero instead of %d' % pyxcd.level

    pyxcd.els()
    pyxcd.w('cdef void mod_cp_rules(')
    pyxcd.ind()
    pyxcd.w('DT_UL_NP_t[:, :] cp_rules,')
    pyxcd.w('DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pyxcd.w('DT_UL_NP_t[:, :] loc_mod_ctr,')
    pyxcd.w('const DT_UL max_idxs_ct,')
    pyxcd.w('const DT_UL n_cps,')
    pyxcd.w('const DT_UL n_pts,')
    pyxcd.w('const DT_UL n_fuzz_nos,')
    pyxcd.w('const DT_UL run_type,')
    pyxcd.w('DT_UL *rand_k,')
    pyxcd.w('DT_UL *rand_i,')
    pyxcd.w('DT_UL *rand_v,')
    pyxcd.w('DT_UL *old_v_i_k,')
    pyxcd.w(') nogil:')

    pxdcd.els()
    pxdcd.w('cdef void mod_cp_rules(')
    pxdcd.ind()
    pxdcd.w('DT_UL_NP_t[:, :] cp_rules,')
    pxdcd.w('DT_UL_NP_t[:, :] cp_rules_idx_ctr,')
    pxdcd.w('DT_UL_NP_t[:, :] loc_mod_ctr,')
    pxdcd.w('const DT_UL max_idxs_ct,')
    pxdcd.w('const DT_UL n_cps,')
    pxdcd.w('const DT_UL n_pts,')
    pxdcd.w('const DT_UL n_fuzz_nos,')
    pxdcd.w('const DT_UL run_type,')
    pxdcd.w('DT_UL *rand_k,')
    pxdcd.w('DT_UL *rand_i,')
    pxdcd.w('DT_UL *rand_v,')
    pxdcd.w('DT_UL *old_v_i_k,')
    pxdcd.w(') nogil')
    pxdcd.ded()

    '''Randomly select a CP, a point and a rule. See if that rule's count is
    not greater than max_idxs_ct. If so, change the rule at the given CP and 
    point.
     
    Takes an initilized cp_rules array using the gen_cp_rules function
    '''
    pyxcd.els()
    pyxcd.w('cdef:')
    pyxcd.ind()
    pyxcd.w('DT_UL dont_stop = 1')
    pyxcd.w('DT_UL max_iters = 1000000, curr_iter_ct = 0')
    pyxcd.w('DT_UL rand_k_, rand_i_, rand_v_, old_v_i_k_')
    pyxcd.ded()

    pyxcd.w('if run_type == 3:')
    pyxcd.ind()
    pyxcd.w('if old_v_i_k[0] < n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('cp_rules_idx_ctr[rand_k[0], old_v_i_k[0]] += 1')
    pyxcd.ded()

    pyxcd.w('if rand_v[0] < n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('cp_rules_idx_ctr[rand_k[0], rand_v[0]] -= 1 ')
    pyxcd.ded()

    pyxcd.w('cp_rules[rand_k[0], rand_i[0]] = old_v_i_k[0]')

    pyxcd.els()
    pyxcd.w('dont_stop = 0  # just in case')
#     pyxcd.w('return')
    pyxcd.ded()

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('while (dont_stop):')
    pyxcd.ind()
    pyxcd.w('curr_iter_ct += 1')
    pyxcd.w('if curr_iter_ct > max_iters:')
    pyxcd.ind()
    pyxcd.w(
        r'printf("\n\n\n\n########Too many iterations in mod_cp_rules!########\n\n\n\n")')
    pyxcd.w('break')
    pyxcd.ded()

    pyxcd.w('rand_k_ = <DT_UL> (rand_c() * n_cps)  # random CP out of n_cps')
    pyxcd.w('rand_i_ = <DT_UL> (rand_c() * n_pts)  # random point in n_pts')

    pyxcd.els()
    pyxcd.w('# random fuzzy rule index out of n_fuzz_nos + 1')
    pyxcd.w('# the extra index is for the points that are supposed to have no ')
    pyxcd.w('# effect on the objective function')
    pyxcd.w('rand_v_ = <DT_UL> (rand_c() * (n_fuzz_nos + 1))')

    pyxcd.w('while ((cp_rules[rand_k_, rand_i_] == rand_v_)):')
    pyxcd.ind()
    pyxcd.w('rand_v_ = <DT_UL> (rand_c() * (n_fuzz_nos + 1))')
    pyxcd.ded()

    pyxcd.w('old_v_i_k_ = cp_rules[rand_k_, rand_i_]')

    pyxcd.els()
    pyxcd.w('if rand_v_ < n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('if cp_rules_idx_ctr[rand_k_, rand_v_] < max_idxs_ct:')
    pyxcd.ind()
    pyxcd.w('cp_rules[rand_k_, rand_i_] = rand_v_')
    pyxcd.w('cp_rules_idx_ctr[rand_k_, rand_v_] += 1')
    pyxcd.els()

    pyxcd.w('if old_v_i_k_ < n_fuzz_nos:')
    pyxcd.ind()
    pyxcd.w('cp_rules_idx_ctr[rand_k_, old_v_i_k_] -= 1')
    pyxcd.ded()

    pyxcd.w('dont_stop = 0')
    pyxcd.ded()
    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('continue')
    pyxcd.ded(lev=2)

    pyxcd.w('else:')
    pyxcd.ind()
    pyxcd.w('cp_rules[rand_k_, rand_i_] = rand_v_')
    pyxcd.w('cp_rules_idx_ctr[rand_k_, old_v_i_k_] -= 1   ')
    pyxcd.w('dont_stop = 0')
    pyxcd.ded(lev=2)

    pyxcd.w('loc_mod_ctr[rand_k_, rand_i_] += 1')

    pyxcd.w('rand_k[0] = rand_k_')
    pyxcd.w('rand_i[0] = rand_i_')
    pyxcd.w('rand_v[0] = rand_v_')
    pyxcd.w('old_v_i_k[0] = old_v_i_k_')
    pyxcd.ded()
    pyxcd.w('return')
    pyxcd.ded()

    assert pyxcd.level == 0, \
        'Level should be zero instead of %d' % pyxcd.level

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

    write_gen_mod_cp_rules_lines(params_dict)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
