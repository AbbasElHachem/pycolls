# cython: linetrace=True, nonecheck=False, boundscheck=False, wrapaound=False

from __future__ import division
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

cdef extern from "math.h" nogil:
    cdef double sin(double x)
    cdef double exp(double x)
    cdef double M_PI
    cdef double pow(double x, double y)


cdef inline double get_dist(double x1, double y1, double x2, double y2) nogil:
    """Get distance between points
    """
    cdef double dist
    dist = pow((((x1 - x2)*(x1 - x2)) + ((y1 - y2)*(y1 - y2))), 0.5)
    return dist


cdef inline double rng_vg(double h, double r, double s) nogil:
    return h


cdef inline double nug_vg(double h, double r, double s) nogil:
    return s


cdef inline double sph_vg(double h, double r, double s) nogil:
    cdef double a, b
    if h >= r:
        return s
    else:
        a = (1.5 * h) / r
        b = (h*h*h) / (2 * (r*r*r))
        return (s * (a - b))


cdef inline double exp_vg(double h, double r, double s) nogil:
    exp_vg = (s * (1 - exp(-h / r)))
    return exp_vg


cdef inline double lin_vg(double h, double r, double s) nogil:
    if h > r:
        return s
    else:
        return s * (h / r)


cdef inline double gau_vg(double h, double r, double s) nogil:
    return (s * (1 - exp(-(((h*h) / (r*r))))))


cdef inline double pow_vg(double h, double r, double s) nogil:
    return (s * (h**r))


cdef inline double hol_vg(double h, double r, double s) nogil:
    cdef double a
    if h == 0:
        return 0
    else:
        a = (M_PI * h) / r
        hol_vg = (s * (1 - (sin(a)/a)))
        return hol_vg


ctypedef double (*f_type)(double h, double r, double s)
cdef map[string, f_type] all_vg_ftns
all_vg_ftns['Rng'] = rng_vg
all_vg_ftns['Nug'] = nug_vg
all_vg_ftns['Sph'] = sph_vg
all_vg_ftns['Exp'] = exp_vg
all_vg_ftns['Lin'] = lin_vg
all_vg_ftns['Gau'] = gau_vg
all_vg_ftns['Pow'] = pow_vg
all_vg_ftns['Hol'] = hol_vg

cdef class OrdinaryKriging:
    '''Do ordinary kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk, est_vars, rhss
        readonly np.ndarray in_dists, in_vars, out_vars, lambdas, mus
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k
        readonly unsigned long in_count, out_count
        readonly vector[double] sills, ranges
        readonly vector[string] vgs, vg_models
        unsigned long f, g, h, i, j, k, l, m, n
        double x1, y1, x_interp, y_interp, out_dist
        double range_f, sill_f
        vector[f_type] vg_ftns
        f_type vg_ftn


    def __init__(self, xi, yi, zi, xk, yk, model='1.0 Sph(2)'):

        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.xk = xk
        self.yk = yk
        self.model = model

        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
        self.mus = np.zeros(shape=(self.out_count,))
        self.est_vars = np.zeros(shape=(self.out_count,))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.in_vars = np.zeros(shape=(self.in_count + 1, self.in_count + 1))
        self.rhss = np.zeros(shape=(self.out_count, self.in_count + 1))


    def krige(self):
        vg_models = self.model.split('+')
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(' ')
            vg, rng = vg.split('(')
            rng = rng.split(')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(float(rng))
            self.vg_ftns.push_back(all_vg_ftns[vg])

        for i in xrange(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in xrange(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1, y1, self.xi[j], self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]

        for f in xrange(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in xrange(self.in_count):
                for g in xrange(self.in_count):
                    if g > h:
                        self.in_vars[h, g] += vg_ftn(self.in_dists[h, g], range_f, sill_f)
                    elif h > g:
                        self.in_vars[h, g] = self.in_vars[g, h]

        self.in_vars[self.in_count, :self.in_count] = np.ones(self.in_count)
        self.in_vars[:, self.in_count] = np.ones(self.in_count + 1)
        self.in_vars[self.in_count, self.in_count] = 0

        for k in xrange(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_vars = np.zeros(shape=(self.in_count + 1))

            for l in xrange(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in xrange(self.vg_ftns.size()):
                    out_vars[l] += self.vg_ftns[n](out_dist, self.ranges[n], self.sills[n])

            out_vars[self.in_count] = 1
            lambdas_k = np.linalg.solve(self.in_vars, out_vars)
            self.mus[k] = lambdas_k[self.in_count]
            lambdas_k = lambdas_k[:self.in_count]
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.est_vars[k] = max(0.0, np.sum(lambdas_k * out_vars[:self.in_count]) + self.mus[k])
            self.lambdas[k, :] = lambdas_k
            self.rhss[k] = out_vars


cdef class SimpleKriging:
    '''Do simple kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk, est_covars, rhss
        readonly np.ndarray in_dists, in_covars, out_covars, lambdas
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k, out_covars_k
        readonly unsigned long in_count, out_count
        readonly vector[double] sills, ranges
        readonly vector[string] vgs, vg_models # cannot access it outside
        unsigned long f, g, h, i, j, k, l, m, n
        double x1, y1, x_interp, y_interp, out_dist
        vector[f_type] vg_ftns
        f_type vg_ftn
        double covar, range_f, sill_f


    def __init__(self, xi, yi, zi, xk, yk, model='1.0 Sph(2)'):

        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.xk = xk
        self.yk = yk
        self.model = model

        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count,))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
        self.est_covars = np.zeros(shape=(self.out_count,))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.covar = np.var(self.zi)
        self.in_covars = np.full(shape=(self.in_count, self.in_count), fill_value=self.covar)
        self.out_covars = np.zeros(shape=(self.in_count, self.in_count))
        self.rhss = np.zeros(shape=(self.out_count, self.in_count))


    def krige(self):
        vg_models = self.model.split('+')
        for submodel in vg_models:
            submodel = submodel.strip()
            sill, vg = submodel.split(' ')
            vg, rng = vg.split('(')
            rng = rng.split(')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(float(rng))
            self.vg_ftns.push_back(all_vg_ftns[vg])

        for i in xrange(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in xrange(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1, y1, self.xi[j], self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]

        for f in xrange(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in xrange(self.in_count):
                for g in xrange(self.in_count):
                    if g > h:
                        self.in_covars[h, g] -= vg_ftn(self.in_dists[h, g], range_f, sill_f)
                    elif h > g:
                        self.in_covars[h, g] = self.in_covars[g, h]

        for k in xrange(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_covars_k = np.full(shape=(self.in_count,), fill_value=self.covar)

            for l in xrange(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in xrange(self.vg_ftns.size()):
                    out_covars_k[l] -= self.vg_ftns[n](out_dist, self.ranges[n], self.sills[n])

            lambdas_k = np.linalg.solve(self.in_covars, out_covars_k)
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.est_covars[k] = max(0.0, self.covar - np.sum(lambdas_k * out_covars_k))
            self.lambdas[k, :] = lambdas_k
            self.out_covars[k, :] = out_covars_k
            self.rhss[k] = out_covars_k


cdef class ExternalDriftKriging:
    '''Do external drift kriging
    '''
    cdef:
        readonly np.ndarray xi, yi, zi, xk, yk, zk
        readonly np.ndarray si, sk, mus_1, mus_2, rhss
        readonly np.ndarray in_dists, in_vars, out_vars, lambdas
        readonly string model, submodel, vg, rng, sill
        np.ndarray lambdas_k
        readonly unsigned long in_count, out_count
        readonly vector[double] sills, ranges
        readonly vector[string] vgs, vg_models
        unsigned long f, g, h, i, j, k, l, m, n
        double x1, y1, x_interp, y_interp, out_dist
        double range_f, sill_f
        vector[f_type] vg_ftns
        f_type vg_ftn


    def __init__(self, xi, yi, zi, si, xk, yk, sk, model='1.0 Sph(2)'):

        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.si = si
        self.xk = xk
        self.yk = yk
        self.sk = sk
        self.model = model

        self.in_count = self.xi.shape[0]
        self.out_count = self.xk.shape[0]
        self.zk = np.zeros(shape=(self.out_count,))
        self.in_dists = np.zeros(shape=(self.in_count, self.in_count))
        self.mus_1 = np.zeros(shape=(self.out_count,))
        self.mus_2 = np.zeros(shape=(self.out_count,))
        self.lambdas = np.zeros(shape=(self.out_count, self.in_count))
        self.in_vars = np.zeros(shape=(self.in_count + 2, self.in_count + 2))
        self.rhss = np.zeros(shape=(self.out_count, self.in_count + 2))

    def krige(self):
        vg_models = self.model.split('+')
        for submodel in vg_models:
            submodel = submodel.strip()
            submodel = submodel.strip()
            sill, vg = submodel.split(' ')
            vg, rng = vg.split('(')
            rng = rng.split(')')[0]
            self.vgs.push_back(vg)
            self.sills.push_back(float(sill))
            self.ranges.push_back(float(rng))
            self.vg_ftns.push_back(all_vg_ftns[vg])

        for i in xrange(self.in_count):
            x1 = self.xi[i]
            y1 = self.yi[i]
            for j in xrange(self.in_count):
                if j > i:
                    self.in_dists[i, j] = get_dist(x1, y1, self.xi[j], self.yi[j])
                elif i > j:
                    self.in_dists[i, j] = self.in_dists[j, i]

        for f in xrange(self.vg_ftns.size()):
            vg_ftn = self.vg_ftns[f]
            range_f = self.ranges[f]
            sill_f = self.sills[f]
            for h in xrange(self.in_count):
                for g in xrange(self.in_count):
                    if g > h:
                        self.in_vars[h, g] += vg_ftn(self.in_dists[h, g], range_f, sill_f)
                    elif h > g:
                        self.in_vars[h, g] = self.in_vars[g, h]

        self.in_vars[self.in_count, :self.in_count] = np.ones(self.in_count)
        self.in_vars[:self.in_count, self.in_count] = np.ones(self.in_count)
        self.in_vars[self.in_count + 1, :self.in_count] = self.si
        self.in_vars[:self.in_count, self.in_count + 1] = self.si

        for k in xrange(self.out_count):
            x_interp = self.xk[k]
            y_interp = self.yk[k]
            out_vars = np.zeros(shape=(self.in_count + 2))

            for l in xrange(self.in_count):
                out_dist = get_dist(self.xi[l], self.yi[l], x_interp, y_interp)
                if out_dist == 0.0:
                    continue
                for n in xrange(self.vg_ftns.size()):
                    out_vars[l] += self.vg_ftns[n](out_dist, self.ranges[n], self.sills[n])

            out_vars[self.in_count] = 1
            out_vars[self.in_count + 1] = self.sk[k]
            lambdas_k = np.linalg.solve(self.in_vars, out_vars)
            self.mus_1[k] = lambdas_k[self.in_count]
            self.mus_2[k] = lambdas_k[self.in_count + 1]
            lambdas_k = lambdas_k[:self.in_count]
            self.zk[k] = np.sum((lambdas_k * self.zi))
            self.lambdas[k, :] = lambdas_k
            self.rhss[k] = out_vars


cdef class OrdinaryIndicatorKriging(OrdinaryKriging):
    '''Do indicator kriging based on ordinary kriging
    '''
    cdef:
        readonly double lim
        unsigned long o
        readonly np.ndarray ik, ixi


    def __init__(self, xi, yi, zi, xk, yk, lim=1, model='1.0 Sph(2)'):
        self.model = model
        OrdinaryKriging.__init__(self, xi, yi, zi, xk, yk, model=self.model)
        self.lim = lim
        self.ixi = np.where(self.zi <= self.lim, 1., 0.)
        self.ik = np.zeros(shape=(self.out_count,) )


    def ikrige(self):
        self.krige()
        for o in xrange(self.out_count):
            self.ik[o] = max(0.0, np.sum((self.lambdas[o, :] * self.ixi)))
            self.est_vars[o] = max(0.0, self.ik[o] * (1. - self.ik[o]))


cdef class SimpleIndicatorKriging(SimpleKriging):
    '''Do indicator kriging based on simple kriging
    '''
    cdef:
        readonly double lim
        unsigned long o
        readonly np.ndarray ik, ixi


    def __init__(self, xi, yi, zi, xk, yk, lim=1, model='1.0 Sph(2)'):
        self.model = model
        SimpleKriging.__init__(self, xi, yi, zi, xk, yk, model=self.model)
        self.lim = lim
        self.ixi = np.where(self.zi <= self.lim, 1., 0.)
        self.ik = np.zeros(shape=(self.out_count,) )


    def ikrige(self):
        self.krige()
        for o in xrange(self.out_count):
            self.ik[o] = max(0.0, np.sum((self.lambdas[o, :] * self.ixi)))
            self.est_covars[o] = max(0.0, self.ik[o] * (1. - self.ik[o]))
