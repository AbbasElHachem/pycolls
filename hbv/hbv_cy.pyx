# cython: nonecheck=False, boundscheck=False, wraparound=False, cdivision=True
from __future__ import division
import time
import sys
import random

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

cdef double MOD_long_long = sys.maxsize * 2.

ctypedef np.float64_t DTYPE_f
ctypedef np.uint32_t DTYPE_i

# start a random number sequence
# xorshift prng
cdef:
    unsigned long long SEED = time.time() * 1000
    unsigned long long rnd_i, A, B, C

rnd_i = SEED
A = 21
B = 35
C = 4

cdef double rand_c() nogil:
    global rnd_i
    rnd_i ^= rnd_i << A
    rnd_i ^= rnd_i >> B
    rnd_i ^= rnd_i << C
    return rnd_i / MOD_long_long

# warmup
cdef unsigned long _
for _ in xrange(1000):
    rand_c()

cdef extern from "math.h" nogil:
    cdef double log(double x)

cdef extern from r'P:\Synchronize\PythonCodes\hbv\hbv_c.h' nogil:
    cdef double loop_HBV_c(const unsigned long *n_recs,
                           const double *conv_ratio, const double *prms_arr,
                           const double *ini_arr, const double *temp_arr,
                           const double *prec_arr, const double *pet_arr,
                           double *q_sim_arr, double *out_arr,
                           const unsigned long *n_ocols)

cdef extern from r'P:\Synchronize\PythonCodes\hbv\c_ftns.h' nogil:
    cdef:
        double get_ns_c(const double *x_arr, double *y_arr,
                        const unsigned long *size,
                        const double *demr,
                        const unsigned long *off_idx)

        double get_ln_ns_c(const double *x_arr, double *y_arr,
                           const unsigned long *size,
                           const double *ln_demr,
                           const unsigned long *off_idx)

        double get_kge_c(const double *act_arr,
                         const double *sim_arr,
                         const double *act_mean,
                         const double *act_std_dev,
                         const unsigned long *size,
                         const unsigned long *off_idx)

cdef void choice_r(const unsigned long[:] x_arr, unsigned long[:] y_arr,
                   const unsigned long n_items, const unsigned long shape) nogil:
    '''get random indicies from x_arr without replacement and put them in y_arr
    '''
    cdef:
        unsigned long i, j, k, l = 1

    while l:
        l = 0
        for i in xrange(n_items):
            k = <unsigned long> (rand_c() * shape)
            y_arr[i] = x_arr[k]

        for i in xrange(n_items):
            for j in xrange(n_items):
                if (i - j) != 0:
                    if y_arr[i] == y_arr[j]:
                        l = 1
    return


cpdef check_choice_r(x_arr, n_items, shape, samples):
    y_arrs = []
    for j in xrange(samples):
        y_arr = np.ones(n_items, dtype='uint32')
        choice_r(x_arr, y_arr, n_items, shape)
        y_arrs.append(y_arr)
    return y_arrs


cdef void del_r(const unsigned long[:] x_arr, unsigned long[:] y_arr,
                const unsigned long idx, const unsigned long shape) nogil:
    '''Delete the element at idx in x_arr and put in y_arr
    '''
    cdef unsigned long i = 0, j = 0

    while (i < shape):
        if i != idx:
            y_arr[i] = x_arr[j]
        else:
            j += 1
            y_arr[i] = x_arr[j]
        i += 1
        j += 1
    return


cpdef check_del_r(const unsigned long[:] x_arr, unsigned long[:] y_arr, const unsigned long idx, const unsigned long shape):
    del_r(x_arr, y_arr, idx, shape)
    return


cdef double obj_ftn(const unsigned long n_recs,
                    const unsigned long off_idx,
                    const double conv_ratio,
                    const double[:] params,
                    const double[:] ini_arr,
                    const double[:] temp_arr,
                    const double[:] prec_arr,
                    const double[:] pet_arr,
                    const double[:] q_act_arr,
                    double[:] q_sim_arr,
                    double[:, :] out_arr,
                    const unsigned long[:] obj_ftn_flags,
                    const double[:] obj_ftn_wts,
                    const double demr,
                    const double ln_demr,
                    const double mean_ref,
                    const double act_std_dev,
                    unsigned long[:] n_calls) nogil:

    cdef:
        double res = 0.0, obj_ftn_wts_sum = 0.0
        unsigned long i, j, n_ocols = out_arr.shape[1]

    if params[4] >= params[2]:
        return 1e4 * (1 + rand_c())

    for i in xrange(n_recs):
        q_sim_arr[i] = 0.0
        for j in xrange(out_arr.shape[1]):
            out_arr[i, j] = 0.0

#    res = loop_HBV_cy(n_recs,
#                      conv_ratio,
#                      params,
#                      ini_arr,
#                      temp_arr,
#                      prec_arr,
#                      pet_arr,
#                      q_sim_arr,
#                      out_arr)

    res = loop_HBV_c(&n_recs,
                     &conv_ratio,
                     &params[0],
                     &ini_arr[0],
                     &temp_arr[0],
                     &prec_arr[0],
                     &pet_arr[0],
                     &q_sim_arr[0],
                     &out_arr[0, 0],
                     &n_ocols)

    if res == 0.0:
        if obj_ftn_flags[0] == 1:
            res += (obj_ftn_wts[0] * get_ns_c(&q_act_arr[0],
                                              &q_sim_arr[0],
                                              &n_recs,
                                              &demr,
                                              &off_idx))
            obj_ftn_wts_sum += obj_ftn_wts[0]
        if obj_ftn_flags[1] == 1:
            res += (obj_ftn_wts[1] * get_ln_ns_c(&q_act_arr[0],
                                                 &q_sim_arr[0],
                                                 &n_recs,
                                                 &ln_demr,
                                                 &off_idx))
            obj_ftn_wts_sum += obj_ftn_wts[1]
        if obj_ftn_flags[2] == 1:
            res += (obj_ftn_wts[2] * get_kge_c(&q_act_arr[0],
                                               &q_sim_arr[0],
                                               &mean_ref,
                                               &act_std_dev,
                                               &n_recs,
                                               &off_idx))
            obj_ftn_wts_sum += obj_ftn_wts[2]

#    if res == 0.0:
#        if obj_ftn_flags[0] == 1:
#            res += (obj_ftn_wts[0] * get_ns(q_act_arr[off_idx:],
#                                            q_sim_arr[off_idx:],
#                                            demr))
#            obj_ftn_wts_sum += obj_ftn_wts[0]
#        if obj_ftn_flags[1] == 1:
#            res += (obj_ftn_wts[1] * get_ln_ns(q_act_arr[off_idx:],
#                                               q_sim_arr[off_idx:],
#                                               ln_demr))
#            obj_ftn_wts_sum += obj_ftn_wts[1]
#        if obj_ftn_flags[2] == 1:
#            res += (obj_ftn_wts[2] * get_kge(q_act_arr[off_idx:],
#                                             q_sim_arr[off_idx:],
#                                             mean_ref, act_std_dev))
#            obj_ftn_wts_sum += obj_ftn_wts[2]

    n_calls[0] = n_calls[0] + 1
    return obj_ftn_wts_sum - res

cpdef dict hbv_de(
    const double[:, :] bounds,
    const unsigned long n_recs,
    const unsigned long off_idx,
    const double conv_ratio,
    const double[:] ini_arr,
    const double[:] temp_arr,
    const double[:] prec_arr,
    const double[:] pet_arr,
    const double[:] q_act_arr,
    const unsigned long[:] obj_ftn_flags,
    const double[:] obj_ftn_wts,
    const double[:] mu_sc_fac_bds,
    const double[:] cr_cnst_bds,
    unsigned long pop_size=1000,
    unsigned long iter_max=1000,
    unsigned long cont_iter_max=30,
    double tol=1e-6):

    '''The differential evolution algorithm for HBV with some modifications
    '''
    cdef:
        unsigned long n_params = bounds.shape[0]
        unsigned long cont_iter = 0

        unsigned long iter_curr = 0
        unsigned long last_succ_i = 0
        unsigned long n_succ = 0

        unsigned long i, j, k
        unsigned long r0, r1, r2

        double mu_sc_fac
        double cr_cnst

        # instead of using tol of present gen use the one from the previous,
            # premature convergence is avoided to some extent through this
        double tol_curr = np.inf, tol_pre = np.inf
        double fval_pre
        double fval_pre_global = np.inf
        double fval_curr

        double mean_ref, ln_mean_ref
        double demr, ln_demr
        double act_std_dev

        np.ndarray[DTYPE_f, ndim=2] pop = np.zeros((pop_size, n_params))
        np.ndarray[DTYPE_i, ndim=1] idx_rng = np.arange(0, pop_size, 1, dtype=np.uint32)
        np.ndarray[DTYPE_f, ndim=1] obj_vals = np.full(pop_size, np.inf, dtype=np.float64)
        np.ndarray[DTYPE_i, ndim=1] del_idx_rng = np.zeros(pop_size - 1, dtype=np.uint32)
        np.ndarray[DTYPE_i, ndim=1] choice_arr = np.zeros(3, dtype=np.uint32)
        np.ndarray[DTYPE_i, ndim=1] params_n_rng = np.arange(0, n_params, 1, dtype=np.uint32)
        np.ndarray[DTYPE_i, ndim=1] r_r = np.zeros(1, dtype=np.uint32)
        np.ndarray[DTYPE_f, ndim=1] best_params = np.full((n_params, ), np.nan)
        np.ndarray[DTYPE_f, ndim=1] v_j_g = np.zeros(n_params)
        np.ndarray[DTYPE_f, ndim=1] u_j_g = np.zeros(n_params)
        np.ndarray[DTYPE_f, ndim=1] params_temp = np.zeros((pop_size,))
        np.ndarray[DTYPE_f, ndim=1] q_sim_arr = np.zeros((n_recs,))
        np.ndarray[DTYPE_f, ndim=2] out_arr = np.zeros(shape=(n_recs + 1, 13), )
        np.ndarray[DTYPE_i, ndim=1] n_calls = np.zeros(1, dtype=np.uint32)

        list idxs_shuff = range(0,  pop_size)
        list accept_vars = []
        list total_vars = []


    mean_ref = get_mean(q_act_arr[off_idx:])
    ln_mean_ref = get_ln_mean(q_act_arr[off_idx:])

    demr = get_demr(q_act_arr[off_idx:], mean_ref)
    ln_demr = get_ln_demr(q_act_arr[off_idx:], ln_mean_ref)

    act_std_dev = get_variance(mean_ref, q_act_arr[off_idx:])**0.5

    # initiate parameter space
    for i in xrange(pop_size):
        for j in xrange(n_params):
            pop[i, j] = bounds[j, 0] + ((bounds[j, 1] - bounds[j, 0]) * \
                        (float(i) / (pop_size - 1)))

    # shuffle the parameters around in the space
    for i in xrange(n_params):
        random.shuffle(idxs_shuff)
        for j in xrange(pop_size):
            params_temp[j] = pop[idxs_shuff[j], i]
        for j in xrange(pop_size):
            pop[j, i] = params_temp[j]

    # for each parameter vector get the objective value
    for i in xrange(pop_size):
        obj_vals[i] = obj_ftn(n_recs, off_idx, conv_ratio, pop[i],
                              ini_arr, temp_arr, prec_arr, pet_arr,
                              q_act_arr, q_sim_arr, out_arr,
                              obj_ftn_flags, obj_ftn_wts, demr, ln_demr,
                              mean_ref, act_std_dev, n_calls)
        if obj_vals[i] < fval_pre_global:
            fval_pre_global = obj_vals[i]
            for k in xrange(n_params):
                best_params[k] = pop[i, k]

#    print 'starting fval_pre_global:', fval_pre_global

#    with nogil: # later
    while (iter_curr < iter_max) and \
          (0.5 * (tol_pre + tol_curr) > tol) and \
          (cont_iter < cont_iter_max):
        cont_iter += 1
#        with nogil, parallel(num_threads=1):
#            for j in prange(pop_size, schedule='dynamic'):

        # randomize the mutation factor for every generation
        #mu_sc_fac = mu_sc_fac_bds[0] + ((mu_sc_fac_bds[1] - mu_sc_fac_bds[0]) * rand_c())

        # randomize the mutation factor for every generation
        #cr_cnst = cr_cnst_bds[0] + ((cr_cnst_bds[1] - cr_cnst_bds[0]) * rand_c())

        for j in range(pop_size):

            # randomize the mutation and recombination factors for every individual of every generation
            mu_sc_fac = mu_sc_fac_bds[0] + ((mu_sc_fac_bds[1] - mu_sc_fac_bds[0]) * rand_c())
            cr_cnst = cr_cnst_bds[0] + ((cr_cnst_bds[1] - cr_cnst_bds[0]) * rand_c())

            # get inidicies except j
            del_r(idx_rng, del_idx_rng, j, pop_size - 1)

            # select indicies randomly from del_idx_rng
            choice_r(del_idx_rng, choice_arr, 3, pop_size - 1)
            r0 = choice_arr[0]
            r1 = choice_arr[1]
            r2 = choice_arr[2]

            # mutate
            for k in xrange(n_params):
                v_j_g[k] = (pop[r0, k] + (mu_sc_fac * (pop[r1, k] - pop[r2, k])))

            # keep parameters in bounds
            for k in xrange(n_params):
                if (v_j_g[k] < bounds[k, 0]) or (v_j_g[k] > bounds[k, 1]):
                    v_j_g[k] = (bounds[k, 0] + ((bounds[k, 1] - bounds[k, 0]) * rand_c()))

            # get an index randomly to have atleast one parameter from the mutated vector
            choice_r(params_n_rng, r_r, 1, n_params)
            for k in xrange(n_params):
                if (rand_c() <= cr_cnst) or (k == r_r[0]):
                    u_j_g[k] = v_j_g[k]
                else:
                    u_j_g[k] = pop[j, k]

            # get objective value for the mutated vector
            fval_curr = obj_ftn(n_recs, off_idx, conv_ratio, u_j_g,
                                ini_arr, temp_arr, prec_arr, pet_arr,
                                q_act_arr, q_sim_arr, out_arr,
                                obj_ftn_flags, obj_ftn_wts, demr, ln_demr,
                                mean_ref, act_std_dev, n_calls)
            fval_pre = obj_vals[j]

            # select if new value is better than the last
            if fval_curr < fval_pre:
                for k in xrange(n_params):
                    pop[j, k] = u_j_g[k]
                obj_vals[j] = fval_curr
                accept_vars.append((mu_sc_fac, cr_cnst, fval_curr, iter_curr))

            total_vars.append((mu_sc_fac, cr_cnst))

            # check for global minimum and best vector
            if fval_curr < fval_pre_global:
                for k in xrange(n_params):
                    best_params[k] = u_j_g[k]
                tol_pre = tol_curr
                tol_curr = (fval_pre_global - fval_curr) / fval_pre_global
                fval_pre_global = fval_curr
                last_succ_i = iter_curr
                n_succ += 1
#                print last_succ_i, fval_pre_global, tol_curr
#                print best_params
                cont_iter = 0

        iter_curr += 1

    # it is important to call the obj_ftn to makes changes one last time
    # i.e. if you want to use/validate results
    obj_ftn(n_recs, off_idx, conv_ratio, best_params,
            ini_arr, temp_arr, prec_arr, pet_arr,
            q_act_arr, q_sim_arr, out_arr,
            obj_ftn_flags, obj_ftn_wts, demr, ln_demr,
            mean_ref, act_std_dev, n_calls)

#    print 'Differential Evolution - Original for HBV:'
#    print 'Number of iterations:', iter_curr
#    print 'Best parameters:', best_params
#    print 'Objective function value:', fval_pre_global
#    print 'Successful tries:', n_succ
#    print 'Last successful try at:', last_succ_i
#    print 'cont_iter:', cont_iter
    return {'params' : best_params,
            'fmin' : fval_pre_global,
            'n_gens': iter_curr,
            'n_succ' : n_succ,
            'lst_succ_try' : last_succ_i,
            'cont_iter' : cont_iter,
            'pop' : pop,
            'fin_tol' : 0.5 * (tol_pre + tol_curr),
            'accept_vars' : accept_vars,
            'total_vars' : total_vars,
            'n_calls' : n_calls[0],
            'pop_obj_vals' : obj_vals}


cdef double get_ns(const double[:] x_arr,
                   const double[:] y_arr,
                   const double demr) nogil:
    '''
    Get Nash - Sutcliffe score
    '''
    cdef:
        unsigned long i
        double numr = 0.0

    for i in xrange(x_arr.shape[0]):
        numr += (x_arr[i] - y_arr[i])**2
    return 1.0 - (numr / demr)


cdef double get_demr(const double[:] x_arr, const double mean_ref) nogil:
    cdef:
        unsigned long i
        double demr = 0.0
    for i in xrange(x_arr.shape[0]):
        demr += (x_arr[i] - mean_ref)**2
    return demr


cdef double get_ln_demr(const double[:] x_arr, const double ln_mean_ref) nogil:
    cdef:
        unsigned long i
        double ln_demr = 0.0
    for i in xrange(x_arr.shape[0]):
        ln_demr += (log(x_arr[i]) - ln_mean_ref)**2
    return ln_demr


cdef double get_ln_mean(const double[:] x_arr) nogil:
    cdef:
        unsigned long i
        double _sum = 0.0

    for i in xrange(x_arr.shape[0]):
        _sum += log(x_arr[i])
    return _sum / (x_arr.shape[0])


cdef double get_ln_ns(const double[:] x_arr,
                      const double[:] y_arr,
                      const double ln_demr) nogil:
    '''
    Get Nash - Sutcliffe score
    '''
    cdef:
        unsigned long i
        double ln_numr = 0.0

    for i in xrange(x_arr.shape[0]):
        ln_numr += (log(x_arr[i]) - \
                    log(y_arr[i]))**2
    return 1.0 - (ln_numr / ln_demr)


cdef inline double get_mean(const double[:] in_arr) nogil:
    cdef:
        double _sum = 0.0
        unsigned long _n = in_arr.shape[0], i = 0

    for i in xrange(_n):
        _sum += in_arr[i]
    return _sum / _n

cdef inline double get_variance(const double in_arr_mean,
                                const double[:] in_arr) nogil:
    cdef:
        double _sum = 0
        unsigned long i, _n = in_arr.shape[0]

    for i in xrange(_n):
        _sum += (in_arr[i] - in_arr_mean)**2
    return _sum / (_n)

cdef inline double get_covar(const double in_arr_1_mean,
                             const double in_arr_2_mean,
                             const double[:] in_arr_1,
                             const double[:] in_arr_2) nogil:
    cdef:
        double _sum = 0
        unsigned long i = 0, _n = in_arr_1.shape[0]

    for i in xrange(_n):
        _sum += ((in_arr_1[i] - in_arr_1_mean) * \
                 (in_arr_2[i] - in_arr_2_mean))
    return _sum / _n


cdef inline double get_correl(const double in_arr_1_std_dev,
                              const double in_arr_2_std_dev,
                              const double arrs_covar) nogil:
    return arrs_covar / (in_arr_1_std_dev * in_arr_2_std_dev)


cdef double get_kge(const double[:] act_arr,
                    const double[:] sim_arr,
                    const double act_mean,
                    const double act_std_dev) nogil:
    cdef:
        double sim_mean, sim_std_dev, covar
        double correl, r, b, g, kge

    sim_mean = get_mean(sim_arr)
    sim_std_dev = get_variance(sim_mean, sim_arr)**0.5

    covar = get_covar(act_mean, sim_mean, act_arr, sim_arr)
    correl = get_correl(act_std_dev, sim_std_dev, covar)

    r = correl
    b = sim_mean / act_mean
    g = (sim_std_dev / sim_mean) / (act_std_dev / act_mean)

    kge = 1 - ((r - 1)**2 + (b - 1)**2 + (g - 1)**2)**0.5
    return kge

cdef double get_corrcoeff(const double[:] x_arr,
                   const double[:] y_arr) nogil:

    cdef:
        double x_mean, y_mean, x_std_dev
        double y_std_dev, covar, correl

    x_mean = get_mean(x_arr)
    y_mean = get_mean(y_arr)

    x_std_dev = get_variance(x_mean, x_arr)**0.5
    y_std_dev = get_variance(y_mean, y_arr)**0.5

    covar = get_covar(x_mean, y_mean, x_arr, y_arr)
    correl = get_correl(x_std_dev, y_std_dev, covar)
    return correl

cpdef fill_correl_mat(double[:, :] vals_arr):
    cdef:
        long i, j
        long shape = vals_arr.shape[1]
        np.ndarray[DTYPE_f, ndim=2] corrs_arr = np.zeros((shape, shape))

    for i in xrange(shape):
        for j in xrange(shape):
            if i > j:
                corrs_arr[i, j] = get_corrcoeff(vals_arr[:, i], vals_arr[:, j])
            elif i == j:
                corrs_arr[i, j] = 1.0

    for i in range(shape):
        for j in range(shape):
            if i < j:
                corrs_arr[i, j] = corrs_arr[j, i]
    return corrs_arr


cdef double loop_HBV_cy(const unsigned long n_recs,
                        const double conv_ratio,
                        const double[:] parms_arr,
                        const double[:] ini_arr,
                        const double[:] temp_arr,
                        const double[:] prec_arr,
                        const double[:] pet_arr,
                        double[:] q_sim_arr,
                        double[:, :] out_arr) nogil:
    cdef:
        DTYPE_f tt = parms_arr[0]
        DTYPE_f c_melt = parms_arr[1]
        DTYPE_f fc = parms_arr[2]
        DTYPE_f beta = parms_arr[3]
        DTYPE_f pwp = parms_arr[4]
        DTYPE_f ur_thresh = parms_arr[5]
        DTYPE_f k_uu = parms_arr[6]
        DTYPE_f k_ul = parms_arr[7]
        DTYPE_f k_d = parms_arr[8]
        DTYPE_f k_ll = parms_arr[9]

        unsigned long i
        double  temp, snow, prec
        double liqu, sm

    out_arr[0, 0] = ini_arr[0]
    out_arr[0, 2] = ini_arr[1]
    out_arr[0, 7] = ini_arr[2]
    out_arr[0, 11] = ini_arr[3]

    if (beta < 1):
        return -1e4 * (2 * rand_c())

    # calculate values
    for i in xrange(1, n_recs + 1):
        temp = temp_arr[(i - 1)]
        snow = out_arr[(i - 1), 0]
        prec = prec_arr[(i - 1)]

        # calc snow and liquid precipitation
        if temp < tt:
            out_arr[i, 0] = snow + prec
            out_arr[i, 1] = 0.
        else:
            out_arr[i, 0] = max(0., snow - c_melt * (temp - tt))
            out_arr[i, 1] = prec + min(snow, c_melt * (temp - tt))

        # calculate evapotranspiration
        liqu = out_arr[i, 1]
        sm = out_arr[(i - 1), 2]
        if sm > pwp:
            out_arr[i, 4] = pet_arr[(i - 1)]
        else:
            out_arr[i, 4] = (sm / fc) * pet_arr[(i - 1)]

        if (sm < 0) or (fc <= 0):
            return -1e4 * (2 * rand_c())

        # soil moisture
        out_arr[i, 2] = sm - out_arr[i, 4] + (liqu * (1 - (sm / fc)**beta))


        # surface runoff
        out_arr[i, 3] = liqu * ((sm / fc)**beta)

        # storage of and discharge from the upper reservoir
        out_arr[i, 8] = max(0.0, k_uu * \
                                 (out_arr[(i - 1), 7] - ur_thresh))

#        out_arr[i, 9] = (out_arr[(i - 1), 7] - out_arr[i, 8]) * k_ul
#        out_arr[i, 10] = max(0.0, (out_arr[(i - 1), 7] - out_arr[i, 9]) * k_d)

        out_arr[i, 10] = (out_arr[(i - 1), 7] - out_arr[i, 8]) * k_d
        out_arr[i, 9] = max(0.0, (out_arr[(i - 1), 7] - out_arr[i, 10]) * k_ul)

#        out_arr[i, 9] = out_arr[(i - 1), 7] * k_ul
#        out_arr[i, 10] = out_arr[(i - 1), 7] * k_d

        out_arr[i, 7] = max(0.0, out_arr[(i - 1), 7] + \
                                 out_arr[i, 3] - \
                                 out_arr[i, 8] - \
                                 out_arr[i, 9] - \
                                 out_arr[i, 10])

        # input to, storage of and discharge from the lower reservoir
        out_arr[i, 12] = out_arr[(i - 1), 11] * k_ll
        out_arr[i, 11] = out_arr[i, 10] + \
                         out_arr[(i - 1), 11] - \
                         out_arr[i, 12]

        out_arr[i, 5] = out_arr[i, 8] + \
                        out_arr[i, 9] + \
                        out_arr[i, 12]
        out_arr[i, 6] = conv_ratio * out_arr[i, 5]
        q_sim_arr[i - 1] = out_arr[i, 6]
    return 0.0


cpdef loop_HBV_py(const unsigned long n_recs,
                  const double conv_ratio,
                  const double[:] params_arr,
                  const double[:] ini_arr,
                  const double[:] temp_arr,
                  const double[:] prec_arr,
                  const double[:] pet_arr):

    cdef:
        np.ndarray[DTYPE_f, ndim=1] q_sim_arr = np.zeros((n_recs,))
        np.ndarray[DTYPE_f, ndim=2] out_arr = np.zeros(shape=(n_recs + 1, 13), )

    loop_HBV_cy(n_recs,
                conv_ratio,
                params_arr,
                ini_arr,
                temp_arr,
                prec_arr,
                pet_arr,
                q_sim_arr,
                out_arr)
    return out_arr[1:, :]


cpdef loop_HBV_cpy(const unsigned long n_recs,
                   const double conv_ratio,
                   const double[:] params_arr,
                   const double[:] ini_arr,
                   const double[:] temp_arr,
                   const double[:] prec_arr,
                   const double[:] pet_arr):

    cdef:
        np.ndarray[DTYPE_f, ndim=1] q_sim_arr = np.zeros((n_recs,))
        np.ndarray[DTYPE_f, ndim=2] out_arr = np.zeros(shape=(n_recs + 1, 13), )
        unsigned long n_ocols = out_arr.shape[1]
        double res

    res = loop_HBV_c(&n_recs,
               &conv_ratio,
               &params_arr[0],
               &ini_arr[0],
               &temp_arr[0],
               &prec_arr[0],
               &pet_arr[0],
               &q_sim_arr[0],
               &out_arr[0, 0],
               &n_ocols)
    return out_arr[1:, :]


cpdef double get_ns_py(const double[:] x_arr, const double[:] y_arr):
    cdef double mean_ref, demr

    mean_ref = get_mean(x_arr)
    demr = get_demr(x_arr, mean_ref)
    return get_ns(x_arr, y_arr, demr)


cpdef double get_ln_ns_py(const double[:] x_arr, const double[:] y_arr):
    cdef double ln_mean_ref, ln_demr

    ln_mean_ref = get_ln_mean(x_arr)
    ln_demr = get_ln_demr(x_arr, ln_mean_ref)
    return get_ln_ns(x_arr, y_arr, ln_demr)


cpdef double get_kge_py(const double[:] x_arr, const double[:] y_arr):
    cdef double mean_ref, act_std_dev

    mean_ref = get_mean(x_arr)
    act_std_dev = get_variance(mean_ref, x_arr)**0.5
    return get_kge(x_arr, y_arr, mean_ref, act_std_dev)


cdef void route_flow(double[:] inflow, double[:] outflow, double lag,
                     double wt, double del_t):
    cdef:
        double C1, C2, C3
        unsigned long i

    C1 = (del_t - (2 * lag * wt)) / \
         ((2 * lag * (1 - wt)) + del_t)
    C2 = (del_t + (2 * lag * wt)) / \
         ((2 * lag * (1 - wt)) + del_t)
    C3 = ((2 * lag * (1 - wt)) - del_t) / \
         ((2 * lag * (1 - wt)) + del_t)

    outflow[0] = inflow[0]
    for i in xrange(1, outflow.shape[0]):
        outflow[i] = (inflow[i] * C1) + \
                     (inflow[i - 1] * C2) + \
                     (outflow[i - 1] * C3)
    return


cpdef route_flow_py(double[:] inflow, double lag, double wt, double del_t):
    cdef:
        np.ndarray[DTYPE_f, ndim=1] outflow = np.zeros((inflow.shape[0],))

    route_flow(inflow, outflow, lag, wt, del_t)
    return outflow