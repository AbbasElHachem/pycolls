# cython: nonecheck=False, boundscheck=False, wraparound=False, cdivision=True
from __future__ import division
import random
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

ctypedef np.float64_t DTYPE_f


cdef extern from 'math.h' nogil:
    cdef double exp(double x)
    cdef double log(double x)
    cdef double M_PI


cdef extern from "c_ftns_norm_cop.h" nogil:
    cdef:
        double get_demr_c(const double *x_arr, const double *mean_ref,
                        const unsigned long *size, const unsigned long *off_idx)

        double get_ln_demr_c(const double *x_arr, const double *ln_mean_ref,
                           const unsigned long *size, const unsigned long *off_idx)

        double get_mean_c(const double *in_arr, const unsigned long *size,
                        const unsigned long *off_idx)

        double get_ln_mean_c(const double *in_arr, const unsigned long *size,
                           const unsigned long *off_idx)

        double get_var_c(const double *in_arr_mean, const double *in_arr,
                       const unsigned long *size, const unsigned long *off_idx)

        double get_ns_c(const double *x_arr, double *y_arr, const unsigned long *size,
                      const double *demr, const unsigned long *off_idx)

        double get_ln_ns_c(const double *x_arr, double *y_arr, const unsigned long *size,
                         const double *ln_demr, const unsigned long *off_idx)

        double get_kge_c(const double *act_arr, const double *sim_arr,
                       const double *act_mean, const double *act_std_dev,
                       const unsigned long *size, const unsigned long *off_idx)

cpdef double get_dist(double x1, double y1, double x2, double y2):
    """Get distance between polongs
    """
    cdef double dist
    dist = (((x1 - x2)**2 + (y1 - y2)**2))**0.5
    return dist


cdef inline double get_mean(double[:] in_arr):
    cdef:
        double _sum = 0
        long _n = in_arr.shape[0], i = 0

    for i in xrange(_n):
        _sum += in_arr[i]
    return _sum / _n


cdef inline double get_variance(double in_arr_mean,
                                double[:] in_arr):
    cdef:
        double _sum = 0
        long i, _n = in_arr.shape[0]

    for i in xrange(_n):
        _sum += (in_arr[i] - in_arr_mean)**2
    return _sum / (_n)


cdef inline double get_covar(double in_arr_1_mean,
                             double in_arr_2_mean,
                             double[:] in_arr_1,
                             double[:] in_arr_2):
    cdef:
        double _sum = 0
        long i = 0, _n = in_arr_1.shape[0]

    for i in xrange(_n):
        _sum += (in_arr_1[i] - in_arr_1_mean) * (in_arr_2[i] - in_arr_2_mean)
    return _sum / _n


cdef inline double get_correl(double in_arr_1_std_dev,
                              double in_arr_2_std_dev,
                              double arrs_covar):
    return arrs_covar / (in_arr_1_std_dev * in_arr_2_std_dev)


cpdef double get_corrcoeff(double[:] act_arr,
                           double[:] sim_arr):

    cdef:
        double act_mean, sim_mean, act_std_dev
        double sim_std_dev, covar, correl

    act_mean = get_mean(act_arr)
    sim_mean = get_mean(sim_arr)

    act_std_dev = get_variance(act_mean, act_arr)**0.5
    sim_std_dev = get_variance(sim_mean, sim_arr)**0.5

    covar = get_covar(act_mean, sim_mean, act_arr, sim_arr)
    correl = get_correl(act_std_dev, sim_std_dev, covar)
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


cpdef double norm_cdf_py(double z, double mu=0.0, double sig=1.0):
    cdef:
        double t, q, p

    z = (z - mu) / sig
    z = z / (2**0.5)

    if z < 0:
        t = 1. / (1. + 0.5 * (-1. * z))
    else:
        t = 1. / (1. + 0.5 * z)

    q = -z**2
    q -= 1.26551223
    q += 1.00002368 * t
    q += 0.37409196 * t**2
    q += 0.09678418 * t**3
    q -= 0.18628806 * t**4
    q += 0.27886807 * t**5
    q -= 1.13520398 * t**6
    q += 1.48851587 * t**7
    q -= 0.82215223 * t**8
    q += 0.17087277 * t**9
    q = exp(q)
    q = q * t

    if z >= 0:
        p = 1 - q
    else:
        p = q - 1

    return 0.5 * (1 + p)


cpdef norm_cdf_py_arr(double[:] z, double mu=0.0, double sig=1.0):
    cdef:
        unsigned long i, n = z.shape[0]
        np.ndarray[DTYPE_f, ndim=1] cdf_arr = np.zeros((n), dtype=np.float64)

    for i in xrange(n):
        cdf_arr[i] = norm_cdf_py(z[i], mu=mu, sig=sig)
    return cdf_arr


cpdef double norm_ppf_py(double p, double mu=0.0, double sig=1.0):
    cdef:
        double t, z

    if p > 0.5:
        t = (-2.0 * log(1 - p))**0.5
    else:
        t = (-2.0 * log(p))**0.5

    z = -0.322232431088 + t * (-1.0 + t * (-0.342242088547 + t * \
        (-0.020423120245 + t * -0.453642210148e-4)))
    z = z / (0.0993484626060 + t * (0.588581570495 + t * \
        (0.531103462366 + t * (0.103537752850 + t * 0.3856070063e-2))))
    z = z + t

    z = (sig * z) + mu
    if p < 0.5:
        return -z
    else:
        return z


cpdef norm_ppf_py_arr(double[:] p, double mu=0.0, double sig=1.0):
    cdef:
        unsigned long i, n = p.shape[0]
        np.ndarray[DTYPE_f, ndim=1] ppf_arr = np.zeros((n), dtype=np.float64)

    for i in xrange(n):
        ppf_arr[i] = norm_ppf_py(p[i], mu=mu, sig=sig)
    return ppf_arr


cpdef double norm_pdf_py(double z, double mu=0.0, double sig=1.0):
    cdef:
        double d

    z = (z - mu) / sig
    d = -0.5 * z**2
    d = exp(d)
    d = d / (2 * M_PI)**0.5
    return d


cpdef norm_pdf_py_arr(double[:] z, double mu=0.0, double sig=1.0):
    cdef:
        unsigned long i, n = z.shape[0]
        np.ndarray[DTYPE_f, ndim=1] pdf_arr = np.zeros((n), dtype=np.float64)

    for i in xrange(n):
        pdf_arr[i] = norm_pdf_py(z[i], mu=mu, sig=sig)
    return pdf_arr


cpdef dict bi_var_copula(double[:] x_probs,
                         double[:] y_probs,
                         unsigned long cop_bins):
    '''get the bivariate empirical copula
    '''

    cdef:
        Py_ssize_t i, j

        unsigned long tot_pts = x_probs.shape[0]

        double div_cnst
        double u1, u2
        unsigned long i_row, j_col

        np.ndarray[DTYPE_f, ndim=2] emp_freqs_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.float64)
        np.ndarray[DTYPE_f, ndim=2] emp_dens_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.float64)

    for i in xrange(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <unsigned long> (u1 * cop_bins)
        j_col = <unsigned long> (u2 * cop_bins)

        emp_freqs_arr[i_row, j_col] += 1

    div_cnst = cop_bins**2 / float(tot_pts)
    for i in xrange(cop_bins):
        for j in xrange(cop_bins):
            emp_dens_arr[i, j] = emp_freqs_arr[i, j] * div_cnst

    return {'emp_freqs_arr': emp_freqs_arr,
            'emp_dens_arr': emp_dens_arr}


cdef double bivar_gau_cop(double t1, double t2, double rho) nogil:
    cdef:
        double cop_dens
    cop_dens = exp(-0.5 * (rho / (1 - rho**2)) * ((rho*(t1**2 + t2**2)) - 2*t1*t2))
    cop_dens /= (1 - rho**2)**0.5
    return cop_dens


cpdef bivar_gau_cop_arr(double rho, unsigned long cop_bins):
    cdef:
        unsigned long i, j
        double p_i, p_j, z_i, z_j
        np.ndarray[DTYPE_f, ndim=2] gau_cop_arr = np.zeros((cop_bins, cop_bins), dtype=np.float64)

    for i in xrange(cop_bins):
        p_i = <double> ((i + 1) / float(cop_bins + 1))
        z_i = norm_ppf_py(p_i)
        for j in xrange(cop_bins):
            p_j = <double> ((j + 1) / float(cop_bins + 1))
            z_j = norm_ppf_py(p_j)
            gau_cop_arr[i, j] = bivar_gau_cop(z_i, z_j, rho)
    return gau_cop_arr


cpdef dict get_rho_tau_from_bivar_emp_dens(double[:] u, double[:, :] emp_dens_arr):
    '''
    Given u, v and frequencies for each u and v, get rho, tau, cummulative
        and Dcummulative values in a dict
    '''
    cdef:
        unsigned long i, j, k, l, rows_cols
        double rho, tau, du, cum_emp_dens, Dcum_emp_dens
        double asymm_1, asymm_2, ui, uj

    rows_cols = u.shape[0]
    du = 1.0 / <double> rows_cols
    dudu = du**2
    rho = 0.0
    tau = 0.0
    asymm_1 = 0.0
    asymm_2 = 0.0

    cdef:
        np.ndarray[DTYPE_f, ndim=2] cum_emp_dens_arr = np.zeros((rows_cols, rows_cols))

    for i in xrange(rows_cols):
        for j in xrange(rows_cols):
            Dcum_emp_dens = emp_dens_arr[i, j] * dudu

            ui = u[i]
            uj = u[j]

            cum_emp_dens = 0.0
            for k in xrange(i + 1):
                for l in xrange(j + 1):
                    cum_emp_dens += emp_dens_arr[k, l]
            cum_emp_dens_arr[i, j] += cum_emp_dens * dudu

            rho += (cum_emp_dens_arr[i, j] - (u[i] * u[j])) * dudu
            tau += cum_emp_dens * Dcum_emp_dens * dudu

            # the old one
#            asymm_1 += ((ui - 0.5) * (uj - 0.5) * (ui + uj - 1) * Dcum_emp_dens)
#            asymm_2 += (-(ui - 0.5) * (uj - 0.5) * (ui - uj) * Dcum_emp_dens)

            # the new one
            asymm_1 += ((ui + uj - 1)**3) * emp_dens_arr[i, j] * dudu
            asymm_2 += ((ui - uj)**3) * emp_dens_arr[i, j] * dudu

    rho = 12.0 * rho
    tau = 4.0 * tau - 1.0

    return {'rho': rho, 'tau': tau, 'asymm_1': asymm_1, 'asymm_2': asymm_2,
            'cumm_dens': cum_emp_dens_arr}


cpdef double tau_sample(double[:] ranks_u, double[:] ranks_v):
    '''Calculate tau_b
    '''
    cdef:
        unsigned long i, j
        unsigned long tie_u = 0, tie_v = 0
        unsigned long n_vals = ranks_u.shape[0]
        double crd, drd, tau, crd_drd, diff_u, diff_v
        double nan = np.nan

    crd = 0.0
    drd = 0.0

    for i in xrange(n_vals):
        for j in xrange(n_vals):
            if i > j:
                diff_u = (ranks_u[i] - ranks_u[j])
                diff_v = (ranks_v[i] - ranks_v[j])
                if diff_u == 0:
                    tie_u += 1
                if diff_v == 0:
                    tie_v += 1

                if (diff_u == 0) and (diff_v == 0):
                    tie_u -= 1
                    tie_v -= 1

                crd_drd = diff_u * diff_v

                if crd_drd > 0:
                    crd += 1
                elif crd_drd < 0:
                    drd += 1

    return (crd - drd) / ((crd + drd + tie_u) * (crd + drd + tie_v))**0.5


cpdef dict get_asymms_sample(double[:] u, double[:] v):
    cdef:
        unsigned long i, n_vals
        double asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in xrange(n_vals):
        asymm_1 += (u[i] + v[i] - 1)**3
        asymm_2 += (u[i] - v[i])**3

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals

    return {'asymm_1':asymm_1, 'asymm_2':asymm_2}


cpdef dict get_asymms_population(double[:] u, double[:, :] emp_dens_arr):
    cdef:
        unsigned long i, j, n_vals
        double asymm_1, asymm_2, dudu

    n_vals = u.shape[0]
    dudu = (u[1] - u[0])**2

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in xrange(n_vals):
        for j in xrange(n_vals):
            asymm_1 += ((u[i] + u[j] - 1)**3) * emp_dens_arr[i, j] * dudu
            asymm_2 += ((u[i] - u[j])**3) * emp_dens_arr[i, j] * dudu

    return {'asymm_1':asymm_1, 'asymm_2':asymm_2}


cpdef get_cond_cumm_probs(double[:] u, double[:, :] emp_dens_arr):
    cdef:
        unsigned long rows
        double du

    rows = u.shape[0]
    du = u[1] - u[0]

    cdef np.ndarray[DTYPE_f, ndim=2] cond_probs = np.zeros((2, rows), dtype=np.float64)
    # cond_probs: zero along the rows, 1 along the columns

    for i in xrange(rows):
        for j in xrange(rows):
            cond_probs[0, i] += emp_dens_arr[i, j]
        cond_probs[0, i] = cond_probs[0, i] * du

    for i in xrange(rows):
        for j in xrange(rows):
            cond_probs[1, i] += emp_dens_arr[j, i]
        cond_probs[1, i] = cond_probs[1, i] * du

    return cond_probs


cpdef double get_ns_py(np.ndarray[double, ndim=1, mode='c'] x_arr,
                       np.ndarray[double, ndim=1, mode='c'] y_arr,
                       const unsigned long off_idx):
    cdef:
        unsigned long size = x_arr.shape[0]
        double mean_ref, demr

    mean_ref = get_mean_c(&x_arr[0], &size, &off_idx)
    demr = get_demr_c(&x_arr[0], &mean_ref, &size, &off_idx)
    return get_ns_c(&x_arr[0], &y_arr[0], &size, &demr, &off_idx)


cpdef double get_ln_ns_py(np.ndarray[double, ndim=1, mode='c'] x_arr,
                          np.ndarray[double, ndim=1, mode='c'] y_arr,
                          const unsigned long off_idx):
    cdef:
        unsigned long size = x_arr.shape[0]
        double ln_mean_ref, ln_demr

    ln_mean_ref = get_ln_mean_c(&x_arr[0], &size, &off_idx)
    ln_demr = get_ln_demr_c(&x_arr[0], &ln_mean_ref, &size, &off_idx)
    return get_ln_ns_c(&x_arr[0], &y_arr[0], &size, &ln_demr, &off_idx)


cpdef double get_kge_py(np.ndarray[double, ndim=1, mode='c'] x_arr,
                        np.ndarray[double, ndim=1, mode='c'] y_arr,
                        const unsigned long off_idx):
    cdef:
        unsigned long size = x_arr.shape[0]
        double mean_ref, act_std_dev

    mean_ref = get_mean_c(&x_arr[0], &size, &off_idx)
    act_std_dev = get_var_c(&mean_ref, &x_arr[0], &size, &off_idx)**0.5
    return get_kge_c(&x_arr[0], &y_arr[0], &mean_ref, &act_std_dev, &size,
                     &off_idx)
