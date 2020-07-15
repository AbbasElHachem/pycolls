# -*- coding: utf-8 -*-
"""
Created on %(14-Jun-16)s

@author: %(Faizan Anwar)s

HBV implementation in cython and python
"""
from __future__ import unicode_literals
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
from pathos.multiprocessing import ProcessPool as mp_pool
from adjustText import adjust_text

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from hbv_cy import (hbv_de, get_ns_py, get_ln_ns_py, loop_HBV_cpy,
                    get_kge_py, loop_HBV_py, fill_correl_mat)

plt.ioff()
np.seterr(all='ignore')
np.set_printoptions(precision=3, linewidth=200)

#TODO: Taking the shape of the catchment to compensate for lag of peak arrivals?
#      Also, how it relates with the temporal resolution
#      Also, change of width along the course of the stream
#      Plot relationship between precipitation and discharge
#      Colored seasons?
#      Add more variables to the top columns in figures,
#      there is space.
#      Get DE parameters' based on input data
class HBVCyPy(object):
    '''The HBV model with some modifications
    '''

    def __init__(self,
                 temperature,
                 precipitation,
                 potential_evapotranspiration,
                 actual_discharge,
                 warm_up_steps,
                 prm_bounds_dict,
                 convert_units_to_input_ratio,
                 obj_ftn_names=['auto'],
                 obj_ftn_wts=[1.0],
                 msgs=True,
                 test_algo=False):

        # get all the input
        self.temp_arr_orig = temperature
        self.prec_arr_orig = precipitation
        self.pet_arr_orig = potential_evapotranspiration
        self.q_act_arr_orig = actual_discharge
        self.off_idx = warm_up_steps
        self.obj_ftn_names = obj_ftn_names
        self.obj_ftn_wts = obj_ftn_wts
        self.conv_ratio = convert_units_to_input_ratio
        self.n_recs_orig = self.temp_arr_orig.shape[0]
        self.msgs = msgs

        self.n_recs = self.n_recs_orig

        self._verify_init_model_01()

        # initial values
        self.ini_snow = 0.0
        self.ini_sm = 0.0
        self.ini_ur_sto = 0.0
        self.ini_lr_sto = 0.0

        if test_algo:
            # fast convergence but local
            self.mu_sc_fac = np.array([0.0, 0.1], dtype='float64') # 0 - 0.1
            self.cr_cnst = np.array([0.9, 1.0], dtype='float64') # 0.9 - 1.0
            self.pop_size = 30 # 30
            self.de_iter_max = 500 # 500
            self.de_max_unsc_iters = 30 # 30
            self.de_tol = 1e-4 # 1e-4
            self.fig_dpi = 60 # 60
        else:
            # slow convergence but global
            self.mu_sc_fac = np.array([0.01, 0.5], dtype='float64') # 0.01 - 0.7
            self.cr_cnst = np.array([0.7, 1.0], dtype='float64') # 0.5 - 1.0
            self.pop_size = 150 # 150
            self.de_iter_max = 1000 # 1000
            self.de_max_unsc_iters = 150 # 150
            self.de_tol = 1e-7 # 1e-9
            self.fig_dpi = 150 # 150

        # for passing to cython
        self.obj_ftn_flags_arr = np.array([0, 0, 0], dtype='uint32')
        self.obj_ftn_wts_arr = np.array([0.0, 0.0, 0.0])

        # specify a dictionary having bounds of each variable
        assert isinstance(prm_bounds_dict, OrderedDict), \
               '\'prm_bounds_dict\' not of type \'OrderedDict\'!'
        self.bounds_dict = OrderedDict()
        self.bounds_dict['tt_bds'] = prm_bounds_dict['tt_bds']
        self.bounds_dict['c_melt_bds'] = prm_bounds_dict['c_melt_bds']
        self.bounds_dict['fc_bds'] = prm_bounds_dict['fc_bds']
        self.bounds_dict['beta_bds'] = prm_bounds_dict['beta_bds']
        self.bounds_dict['pwp_bds'] = prm_bounds_dict['pwp_bds']
        self.bounds_dict['ur_thresh_bds'] = prm_bounds_dict['ur_thresh_bds']
        self.bounds_dict['k_uu_bds'] = prm_bounds_dict['k_uu_bds']
        self.bounds_dict['k_ul_bds'] = prm_bounds_dict['k_ul_bds']
        self.bounds_dict['k_d_bds'] = prm_bounds_dict['k_d_bds']
        self.bounds_dict['k_ll_bds'] = prm_bounds_dict['k_ll_bds']

        # get the objective functions
        self.n_obj_ftns = 0

        if 'auto' in self.obj_ftn_names:
            avg_flow = np.mean(self.q_act_arr_orig)
            flow_ge = np.where(self.q_act_arr_orig > avg_flow, 1, 0)
            n_flow_ge = np.sum(flow_ge)
            ns_ratio = n_flow_ge / float(self.q_act_arr_orig.shape[0])
            self.obj_ftn_names = ['ns', 'ln_ns']
            self.obj_ftn_wts = [1 - ns_ratio, ns_ratio]
            self.n_obj_ftns = 2

            self.obj_ftn_flags_arr = np.array([1, 1, 0], dtype='uint32')
            self.obj_ftn_wts_arr = np.array(np.concatenate((self.obj_ftn_wts, [0.0])),
                                            dtype='float64')
        else:
            if 'ns' in self.obj_ftn_names:
                self.n_obj_ftns += 1
                self.obj_ftn_flags_arr[0] = 1
                self.obj_ftn_wts_arr[0] = self.obj_ftn_wts[self.obj_ftn_names.index('ns')]
            if 'ln_ns' in self.obj_ftn_names:
                self.n_obj_ftns += 1
                self.obj_ftn_flags_arr[1] = 1
                self.obj_ftn_wts_arr[1] = self.obj_ftn_wts[self.obj_ftn_names.index('ln_ns')]
            if 'kge' in self.obj_ftn_names:
                self.n_obj_ftns += 1
                self.obj_ftn_flags_arr[2] = 1
                self.obj_ftn_wts_arr[2] = self.obj_ftn_wts[self.obj_ftn_names.index('kge')]

        self.param_bds_syms = ['tt_bds', 'c_melt_bds', 'fc_bds',
                               'beta_bds', 'pwp_bds', 'ur_thresh_bds',
                               'k_uu_bds', 'k_ul_bds', 'k_d_bds', 'k_ll_bds']

        self._verify_init_model_02()

        self.obj_ftn_str = ''
        for q in range(self.n_obj_ftns):
            self.obj_ftn_str += '%0.3f * %s + ' % (self.obj_ftn_wts[q], self.obj_ftn_names[q])
        self.obj_ftn_str = self.obj_ftn_str[:-3]

        self.simple_opt_flag = False
        self.kfold_opt_flag = False
        self.npop_opt_flag = False
        self.de_pool = None
        self.debug_mode = False
        self.params_npops_df = None
        self.nprocs = 1

        # order is important
        self.param_names = ['Threshold Temperature', 'C_melt', 'Field Capacity', 'Beta',
                            'Permanent Wilting Point', 'ur_thresh', 'K_u', 'K_l', 'K_d', 'K_ll']
        self.param_syms = ['TT', 'C_melt', 'FC', 'Beta',
                           'PWP', 'ur_thresh', 'K_u', 'K_l', 'K_d', 'K_ll']
        self.obj_strs = ['obj_ftn_str', 'fmin_cy', 'fmin_py', 'ns', 'ln_ns', 'kge', 'run_type']
        self.other_syms = ['strt_idx', 'end_idx', 'conv_fac', 'ini_snow', 'ini_sm', 'ini_ur_sto', 'ini_lr_sto']
        self.de_stats = ['n_gens', 'n_succ', 'lst_succ_try', 'cont_iter', 'fin_tol', 'tot_calls']
        self.stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
        return

    def _verify_init_model_01(self):
        in_arrs_list = [self.temp_arr_orig, self.prec_arr_orig,
                        self.pet_arr_orig, self.q_act_arr_orig]
        for arr in in_arrs_list:
            assert arr.shape[0] == in_arrs_list[0].shape[0], \
                   'Input vectors\' shapes are unequal!'
            assert len(arr.shape) == 1, \
                   'Input variable arrays are not column vectors!'
            assert np.all(np.isfinite(arr)), 'Infinite values in input vectors!'
            assert arr.dtype.name == 'float64', 'Data type of input variable array not float64!'
        assert not np.any(np.where(self.q_act_arr_orig == 0, True, False)), \
               'No zeros allowed in actual discharge array!'

        assert isinstance(self.off_idx, int), 'Parameter \'warm_up_steps\' not an integer!'
        assert self.off_idx >= 0, 'Parameter \'warm_up_steps\' cannot be less than zero!'

        assert hasattr(self.obj_ftn_names, '__iter__'), 'Parameter \'obj_ftn_names\' not an iterable!'
        assert hasattr(self.obj_ftn_wts, '__iter__'), 'Parameter \'obj_ftn_wts\' not an iterable!'

        assert len(self.obj_ftn_names) == len(self.obj_ftn_wts), \
               'Unequal number of objective function names and weights!'

        assert isinstance(self.conv_ratio, float), 'Parameter \'convert_units_to_input_ratio\' not a float!'
        assert self.conv_ratio > 0, 'Parameter \'convert_units_to_input_ratio\' cannot be less than or equal to zero!'
        assert np.isfinite(self.conv_ratio), 'Parameter \'convert_units_to_input_ratio\' is infinite or NaN!'
        return

    def _verify_init_model_02(self):
        self.bounds_arr = np.array(self.bounds_dict.values(), dtype='float64')
        assert self.bounds_arr.shape[0] == len(self.param_bds_syms), \
               ('Supplied bounds of parameters are not %d!' % len(self.param_bds_syms))
        assert np.all(np.isfinite(self.bounds_arr)), \
               'Invalid values of model parameters\' bounds!'
        assert len(self.bounds_arr.shape) == 2, \
               'More than two values specified as parameter bounds!'
        assert np.all(np.where(self.bounds_arr[:, 1] - self.bounds_arr[:, 0] >= 0, True, False)), \
               'Model bounds not in ascending order!'
        assert self.param_bds_syms == self.bounds_dict.keys(), \
               '\'bounds_dict\'s\' keys not in required order. Should be: %s' % str(self.param_bds_syms)
        assert (len(self.obj_ftn_names) != 0) and (len(self.obj_ftn_wts) != 0), \
               'Objective function names and weights are inconsistent!'
        assert (len(self.obj_ftn_names) == self.n_obj_ftns) and \
               (len(self.obj_ftn_wts) == self.n_obj_ftns), \
               'Unequal number of objective function names and weights!'

        ini_vals_list = [self.ini_snow,
                         self.ini_sm,
                         self.ini_ur_sto,
                         self.ini_lr_sto]

        for ini_val in ini_vals_list:
            assert isinstance(ini_val, float), 'Initial value of model parameter not a float!'
            assert ini_val >= 0, 'Initial value of model parameter cannot be less than zero!'

        # to change the initial conditions:
            # assign the initial values manually from outside
            # these will be added to the ini_arr before optimization
        self.ini_arr = np.array(ini_vals_list, dtype='float64')
        return

    def _verify_de_params(self):
        assert isinstance(self.mu_sc_fac, np.ndarray), '\'mu_sc_fac\' is not a np.ndarray object!'
        assert isinstance(self.cr_cnst, np.ndarray), '\'mu_sc_fac\' is not a np.ndarray object!'

        assert self.mu_sc_fac.dtype.name == 'float64', '\'mu_sc_fac\'s\' dtype is not float64!'
        assert self.cr_cnst.dtype.name == 'float64', '\'cr_cnst\'s\' dtype is not float64!'

        assert self.mu_sc_fac.shape[0] == self.cr_cnst.shape[0] == 2, \
               'Two bounds required for mutation and recombination factor!'
        assert len(self.mu_sc_fac.shape) == len(self.cr_cnst.shape) == 1, \
               'Mutation factor and recombination bounds arrays have more than one-dimension!'

        assert self.mu_sc_fac[0] <= self.mu_sc_fac[1], 'Mutation factor array not ascending!'
        assert self.cr_cnst[0] <= self.cr_cnst[1], 'Recombination array not ascending!'

        assert isinstance(self.pop_size, int), 'Poulation size not an integer!'
        assert isinstance(self.de_iter_max, int), 'Maximum iterations not an integer!'
        assert isinstance(self.de_max_unsc_iters, int), \
               'Continuous maximum unsuccessful iterations not an integer!'
        assert isinstance(self.de_tol, float), 'Termination tolerance not a float!'

        assert self.pop_size > 0, 'Population size cannot be less than 1!'
        assert self.de_iter_max > 0, 'Maximum iterations cannot be less than one!'
        assert self.de_max_unsc_iters > 0, \
               'Continuous maximum unsuccessful iterations cannot be less than 1!'
        assert self.de_tol > 0, 'Termination tolerance cannot be equal or less than zero!'
        assert np.isfinite(self.de_tol), 'Termination tolerance cannot be NaN or infinite!'
        return

    def _start_pool(self):
        '''
        call it in the first line of the function that has mp in it
        '''
        if self.debug_mode:
            self.nprocs = 1
        elif not hasattr(self.de_pool, 'ncpus'):
            self.de_pool = mp_pool(nodes=self.nprocs)
        return

    def _reassign_arrs(self, idx_01, idx_02):
        self.temp_arr = self.temp_arr_orig[idx_01:idx_02].copy()
        self.prec_arr = self.prec_arr_orig[idx_01:idx_02].copy()
        self.pet_arr = self.pet_arr_orig[idx_01:idx_02].copy()
        self.q_act_arr = self.q_act_arr_orig[idx_01:idx_02].copy()
        self.n_recs = self.pet_arr.shape[0]
        return

    def _reset_params(self):
        '''At the start of each optimization function call this

        It resets all the stuff so that you don't use any output
        from another function.
        '''
        self._reassign_arrs(0, self.n_recs_orig + 1)

        self.params_arr = 10 * [None]
        self.tt, \
        self.c_melt, \
        self.fc, \
        self.beta, \
        self.pwp, \
        self.ur_thresh, \
        self.k_uu, \
        self.k_ul, \
        self.k_d, \
        self.k_ll = self.params_arr

        self.snow_arr = None
        self.liqu_arr = None
        self.sm_arr = None
        self.tot_run_arr = None
        self.evap_arr = None
        self.comb_run_arr = None
        self.q_sim_arr = None
        self.ur_sto_arr = None
        self.ur_run_uu = None
        self.ur_run_ul = None
        self.ur_to_lr_run = None
        self.lr_sto_arr = None
        self.lr_run_arr = None

        self.npops = None
        self.nprocs = None
        self.opt_multi_pop_res = None

        self.kfolds = None
        self.idxs_list = None
        self.calib_valid_arr = None
        self.ns_arr = None
        self.ln_ns_arr = None
        self.kge_arr = None
        self.params_kfolds_df = None

        self.params_npops_df = None
        self.best_params_arr = None

        self.out_multi_pop_dir = None
        self.out_multi_pop_suff = None

        self.save_hbv = None
        self.n_params = len(self.param_syms)
        self.n_stats_cols = len(self.stats_cols)

        self.n_rows, self.n_cols = 5, 2
        self.tick_font_size = 7
        self.hist_bins = 20

        self.simple_opt_flag = False
        self.kfold_opt_flag = False
        self.npop_opt_flag = False
        return

    @property
    def ns(self):
        '''
        Get Nash - Sutcliffe score
        '''
        ns = get_ns_py(self.q_act_arr[self.off_idx:], self.q_sim_arr[self.off_idx:])
        return ns

    @property
    def ln_ns(self):
        '''
        Get ln Nash - Sutcliffe score
        '''
        ln_ns = get_ln_ns_py(self.q_act_arr[self.off_idx:], self.q_sim_arr[self.off_idx:])
        return ln_ns

    @property
    def kge(self):
        '''
        Get Kling - Gupta Efficiency
        '''
        kge = get_kge_py(self.q_act_arr[self.off_idx:], self.q_sim_arr[self.off_idx:])
        return kge

    @property
    def obj_val_py(self):
        '''Get a weighted objective value using the specified efficiency functions
        '''
        obj_val = np.array([(1 - self.ns), (1 - self.ln_ns), (1 - self.kge)])
        obj_val = np.sum(self.obj_ftn_flags_arr * self.obj_ftn_wts_arr * obj_val)
        return obj_val

    def optimize(self):
        '''Optimize the HBV model one time
        '''

        if self.msgs:
            print '\nOptimizing using the entire input...'
            print 'Initial conditions are:\n', self.ini_arr

        self._reset_params()
        self._verify_init_model_01()
        self._verify_init_model_02()
        self._verify_de_params()

        opt_res = self._get_opt_mp(0)
        self.opt_multi_pop_res = [opt_res]
        self._chk_nans_in_pop()

        self.tt, \
        self.c_melt, \
        self.fc, \
        self.beta, \
        self.pwp, \
        self.ur_thresh, \
        self.k_uu, \
        self.k_ul, \
        self.k_d, \
        self.k_ll = opt_res['params']
        self.params_arr = opt_res['params']

        self._run_sim()

        if self.msgs:
            print 'Used objective functions:', self.obj_ftn_names
            print 'Objective functions weights:', self.obj_ftn_wts
            print 'Objective function value from cython:', opt_res['fmin']
            print 'Objective function value from python:', self.obj_val_py
            print 'Number of generations:', opt_res['n_gens']
            print 'Number of successful tries:', opt_res['n_succ']
            print 'Last successful try at:', opt_res['lst_succ_try']
            print 'Continuous iterations without success:', opt_res['cont_iter']
            print 'Final tolerance:', opt_res['fin_tol']
            print 'Total calls to loop_HBV:', opt_res['n_calls']

            print '   NS is:', self.ns
            print 'Ln NS is:', self.ln_ns
            print '  KGE is:', self.kge

            print 'Optimized parameters are:'
            for i, prm in enumerate(self.params_arr):
                print '%23s: %0.3f' % (self.param_names[i], prm)

            print 'Done optimizing.'

        self.npops = 1
        self.nprocs = 1
        self.simple_opt_flag = True
        return

    def optimize_kfolds(self, kfolds, nprocs=1):
        '''Perform calibration and validation by dividing data into kfolds
        '''
        self._reset_params()

        self.kfolds = kfolds
        assert isinstance(self.kfolds, int), 'Parameter \'kfolds\' not an integer!'
        assert self.kfolds > 1, 'Paramter \'kfolds\' cannot be less than 2!'

        if nprocs == 'auto':
            self.nprocs = int(os.environ["NUMBER_OF_PROCESSORS"]) - 1
        else:
            self.nprocs = nprocs

        self.idxs_list = pd.np.linspace(0, self.n_recs_orig, self.kfolds + 1, endpoint=True, dtype='int64')
        if self.idxs_list.shape[0] == 1:
            self.idxs_list = pd.np.concatenate((pd.np.array([0], dtype='int64'), self.idxs_list))
            self.nprocs = 1

        self.calib_valid_arr = np.full(shape=(self.kfolds, self.kfolds), fill_value=np.nan)
        self.ns_arr = self.calib_valid_arr.copy()
        self.ln_ns_arr = self.calib_valid_arr.copy()
        self.kge_arr = self.calib_valid_arr.copy()

        self._verify_init_model_01()
        self._verify_init_model_02()
        self._verify_de_params()

        self.npops = self.kfolds

        if self.msgs:
            print '\nOptimizing using kfolds....'
            print 'Initial conditions are:\n', self.ini_arr

        cols_list = []
        cols_list.extend(self.param_syms)
        cols_list.extend(self.obj_strs)
        cols_list.extend(self.other_syms)
        self.params_kfolds_df = pd.DataFrame(index=range(0, self.kfolds**2), columns=cols_list)

        self._start_pool()

        if (self.nprocs == 1) or self.debug_mode:
            self.opt_multi_pop_res = []
            for n in xrange(self.kfolds):
                self.opt_multi_pop_res.append(self._get_kfold_opt_mp(n))
        else:
            self.opt_multi_pop_res = self.de_pool.map(self._get_kfold_opt_mp, range(self.kfolds))
            self.de_pool.clear()

        self._chk_nans_in_pop()

        ctr = 0
        for n in range(self.kfolds):
            opt_res = self.opt_multi_pop_res[n]

            for o in range(self.kfolds):
                self._reassign_arrs(self.idxs_list[o], self.idxs_list[o + 1])
                self.q_sim_arr = loop_HBV_py(self.n_recs,
                                             self.conv_ratio,
                                             opt_res['params'],
                                             self.ini_arr,
                                             self.temp_arr,
                                             self.prec_arr,
                                             self.pet_arr)[:, 6]

                self.calib_valid_arr[n, o] = self.obj_val_py
                self.ns_arr[n, o] = self.ns
                self.ln_ns_arr[n, o] = self.ln_ns
                self.kge_arr[n, o] = self.kge

                if n == o:
                    runtype = 'calibration'
                else:
                    runtype = 'validation'

                params_list = [float(p) for p in opt_res['params']]
                params_list.extend([self.obj_ftn_str, opt_res['fmin'],
                                    self.calib_valid_arr[n, o],
                                    self.ns_arr[n, o], self.ln_ns_arr[n, o],
                                    self.kge_arr[n, o], runtype, self.idxs_list[o],
                                    self.idxs_list[o + 1], self.conv_ratio])
                params_list.extend([float(ini) for ini in self.ini_arr])
                self.params_kfolds_df.iloc[ctr] = params_list
                ctr += 1

        if self.msgs:
            print 'Done optimizing using kfolds.'
        self.kfold_opt_flag = True
        return

    def optimize_multi_pop(self, nprocs, npops):
        '''Optimize based on given number of populations

        This is just for testing that how restarting a simulation affects
        the parameters.
        '''
        self._reset_params()

        if nprocs == 'auto':
            self.nprocs = int(os.environ["NUMBER_OF_PROCESSORS"]) - 1
        else:
            self.nprocs = nprocs

        self.npops = npops

        assert isinstance(self.nprocs, int), 'Number of processes not an integer!'
        assert isinstance(self.npops, int), 'Number of populations not an integer!'

        assert self.nprocs > 0, 'Number of processes cannot be less than 1!'
        assert self.npops > 0, 'Number of populations cannot be less than 1!'

        self._start_pool()

        if self.msgs:
            print '\nOptimizing using multiple populations....'
            print 'Initial conditions are:\n', self.ini_arr

        self._verify_init_model_01()
        self._verify_init_model_02()
        self._verify_de_params()

        if (self.nprocs == 1) or self.debug_mode:
            self.opt_multi_pop_res = []
            for i in xrange(self.npops):
                self.opt_multi_pop_res.append(self._get_opt_mp(i))
        else:
            self.opt_multi_pop_res = self.de_pool.map(self._get_opt_mp, range(self.npops))
            self.de_pool.clear()

        self._chk_nans_in_pop()

        cols_list = self.de_stats
        cols_list.extend(self.obj_strs[:-1])
        cols_list.extend(self.param_syms)

        self.params_npops_df = pd.DataFrame(index=range(0, self.npops), columns=cols_list)

        if self.msgs:
            print 'Used objective functions:', self.obj_ftn_names
            print 'Objective functions weights:', self.obj_ftn_wts

        for i, opt_res in enumerate(self.opt_multi_pop_res):
            self.q_sim_arr = loop_HBV_py(
                             self.n_recs_orig,
                             self.conv_ratio,
                             opt_res['params'],
                             self.ini_arr,
                             self.temp_arr_orig,
                             self.prec_arr_orig,
                             self.pet_arr_orig)[:, 6]

            res_list = []
            res_list.extend([opt_res['n_gens'], opt_res['n_succ'], opt_res['lst_succ_try']])
            res_list.extend([opt_res['cont_iter'], opt_res['fin_tol']])
            res_list.extend([(opt_res['n_gens'] * self.pop_size) + self.pop_size + 1])
            res_list.extend([self.obj_ftn_str, opt_res['fmin'], self.obj_val_py])
            res_list.extend([self.ns, self.ln_ns, self.kge])
            res_list.extend([x for x in opt_res['params']])

            self.params_npops_df.iloc[i] = res_list

        self.best_params_arr = np.array(self.params_npops_df.values[:, -13:], dtype='float64')

        if self.msgs:
            prms_strs = ['NS', 'Ln_NS', 'KGE', 'Threshold Temperature', 'C_melt', 'Field Capacity', 'Beta',
                         'Permanent Wilting Point', 'ur_thresh', 'K_u', 'K_l', 'K_d', 'K_ll']
            params_mins = ['%0.4f' % x for x in np.min(self.best_params_arr, axis=0)]
            params_maxs = ['%0.4f' % x for x in np.max(self.best_params_arr, axis=0)]

            print 'Min and Max values of params:'
            for i, prm in enumerate(zip(params_mins, params_maxs)):
                print '%23s: %s, %s' % (prms_strs[i], prm[0], prm[1])

        if self.msgs:
            print 'Done optimizing using multiple populations.'
        self.npop_opt_flag = True
        return

    def run_sim(self, temp_arr_orig, prec_arr_orig, pet_arr_orig, params_arr=None):
        '''Run a simulation based on given data from user
        '''
        if self.msgs:
            print 'Running based on given input simulation...'

        self.n_recs_orig = temp_arr_orig.shape[0]
        self.temp_arr_orig = temp_arr_orig
        self.prec_arr_orig = prec_arr_orig
        self.pet_arr_orig = pet_arr_orig

        self._verify_init_model_01()
        self._verify_init_model_02()

        if params_arr:
            self.params_arr = params_arr
            assert np.all(np.isfinite(self.params_arr)), 'NaN or infinite values in \'params_arr\'!'
            assert self.params_arr.dtype.name == 'float64', 'Data type of \'params_arr\' not float64!'

        all_output = loop_HBV_py(
                     self.n_recs_orig,
                     self.conv_ratio,
                     self.params_arr,
                     self.ini_arr,
                     temp_arr_orig,
                     prec_arr_orig,
                     pet_arr_orig)

        self.snow_arr = all_output[:, 0]
        self.liqu_arr = all_output[:, 1]
        self.sm_arr = all_output[:, 2]
        self.tot_run_arr = all_output[:, 3]
        self.evap_arr = all_output[:, 4]
        self.comb_run_arr = all_output[:, 5]
        self.q_sim_arr = all_output[:, 6]
        self.ur_sto_arr = all_output[:, 7]
        self.ur_run_uu = all_output[:, 8]
        self.ur_run_ul = all_output[:, 9]
        self.ur_to_lr_run = all_output[:, 10]
        self.lr_sto_arr = all_output[:, 11]
        self.lr_run_arr = all_output[:, 12]

        if self.msgs:
            print 'Done running simulation.'
        self.simple_opt_flag = True
        return

    # TODO: Implement this
    def save_sim(self):
        return

    def load_params(self, in_params_loc):
        in_param_df = pd.read_csv(in_params_loc, sep=str(';'), index_col=0)
        if in_param_df.index.tolist()[:-3] != self.param_syms:
            raise TypeError('The supplied input dataframe is incorrect!')
        else:
            self.params_arr = in_param_df['value'].values[:-3]
            self.tt, \
            self.c_melt, \
            self.fc, \
            self.beta, \
            self.pwp, \
            self.ur_thresh, \
            self.k_uu, \
            self.k_ul, \
            self.k_d, \
            self.k_ll = self.params_arr
        return

    def save_simple_opt(self, out_dir, out_suff, save_de=False):
        '''Save the output of the optimize function
        '''
        assert self.simple_opt_flag or self.npop_opt_flag or self.kfold_opt_flag, \
               'Optimize first!'

        self._verify_init_model_01()
        self._reassign_arrs(0, self.n_recs_orig + 1)

        assert np.any(self.params_arr), 'HBV parameters are not defined!'
        assert self.q_act_arr_orig.shape[0] == self.q_sim_arr.shape[0], \
               'Original and simulated discharge have unequal steps!'

        if self.msgs:
            print '\nSaving output figure and parameters file...'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_fig_loc = os.path.join(out_dir, 'HBV_model_plot_%s.png' % out_suff)
        out_params_loc = os.path.join(out_dir, 'HBV_model_params_%s.csv' % out_suff)

        out_labs = []
        out_labs.extend(self.param_syms)
        out_labs.extend(['ns', 'ln_ns', 'kge', 'obj_ftn'])

        out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
        out_params_df['value'] = np.concatenate((self.params_arr, [self.ns, self.ln_ns, self.kge, self.obj_ftn_str]))
        out_params_df.to_csv(out_params_loc, sep=str(';'), index_label='param')


        cum_q_orig = np.cumsum(self.q_act_arr_orig[self.off_idx:]) / self.conv_ratio
        cum_q_sim = np.cumsum(self.comb_run_arr[self.off_idx:])
        vol_diff_arr = np.concatenate((np.full(shape=(self.off_idx - 1), fill_value=np.nan), cum_q_sim / cum_q_orig))
#        vol_diff_arr = np.concatenate((np.zeros(shape=(self.off_idx - 1)),
#                                       np.cumsum((self.q_act_arr_orig[self.off_idx:] / self.conv_ratio) - \
#                                                  self.comb_run_arr[self.off_idx:])))

        bal_idxs = [0]
        bal_idxs.extend(range(self.off_idx, self.n_recs_orig, 365 * 2))
        act_bal_arr = []
        sim_bal_arr = []
        #TODO: plot with and without but in a seperate function
        for i in range(1, len(bal_idxs) - 1):
            # ET accounted for
            act_bal_arr.append((np.sum(self.q_act_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) / self.conv_ratio) / \
                               (np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) - \
                                np.sum(self.evap_arr[bal_idxs[i]:bal_idxs[i + 1]])))

            sim_bal_arr.append(np.sum(self.comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]]) / \
                              (np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) - \
                               np.sum(self.evap_arr[bal_idxs[i]:bal_idxs[i + 1]])))

            # ET not accounted for
#            act_bal_arr.append((np.sum(self.q_act_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) / self.conv_ratio) / \
#                               np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]))
#
#            sim_bal_arr.append(np.sum(self.comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]]) / \
#                               np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]))

        act_bal_arr = np.concatenate(([np.nan, np.nan], act_bal_arr), axis=0)
        sim_bal_arr = np.concatenate(([np.nan, np.nan], sim_bal_arr), axis=0)

        plt.figure(figsize=(30, 50))
        t_rows = 16
        t_cols = 10

        i = 0
        params_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        plt.title('HBV Flow Simulation')
        discharge_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        vol_err_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        balance_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        prec_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        liqu_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        snow_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        temp_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        pet_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        sm_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        tot_run_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        u_res_sto_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        ur_run_uu_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        ur_run_ul_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        ur_to_lr_run_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        l_res_sto_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)

        bar_x = np.arange(0, self.n_recs_orig, 1)
        discharge_ax.plot(self.q_act_arr_orig, 'r-', label='Actual', lw=0.8)
        discharge_ax.plot(self.q_sim_arr, 'b-', label='Simulated', lw=0.5, alpha=0.5)
        vol_err_ax.axhline(1.0, color='k')
        vol_err_ax.plot(vol_diff_arr, lw=0.5, label='Cumm. Runoff Error')
        vol_err_ax.set_ylim(0.5, 1.5)
        balance_ax.axhline(1.0, color='k')
        balance_ax.scatter(bal_idxs, act_bal_arr, c='r', label='Actual Volume', alpha=0.6)
        balance_ax.scatter(bal_idxs, sim_bal_arr, c='b', label='Simulated Volume', alpha=0.6)
        balance_ax.set_xlim(0, discharge_ax.get_xlim()[1])
        balance_ax.set_ylim(0.5, 1.5)
        prec_ax.bar(bar_x, self.prec_arr_orig, label='Precipitation', edgecolor='none', width=1.0)
        temp_ax.plot(self.temp_arr_orig, lw=0.5, label='Temperature')
        pet_ax.plot(self.pet_arr_orig, 'r-', lw=0.8, label='Potential Evapotranspiration')
        pet_ax.plot(self.evap_arr, 'b-', lw=0.5, label='Evapotranspiration', alpha=0.5)
        snow_ax.plot(self.snow_arr, lw=0.5, label='Snow')
        liqu_ax.bar(bar_x, self.liqu_arr, width=1.0, edgecolor='none', label='Liquid Precipitation')
        sm_ax.plot(self.sm_arr, lw=0.5, label='Soil Moisture')
        tot_run_ax.bar(bar_x, self.tot_run_arr, width=1.0, label='Total Runoff', edgecolor='none')
        u_res_sto_ax.plot(self.ur_sto_arr, lw=0.5, label='Upper Reservoir - Storage')
        ur_run_uu_ax.bar(bar_x, self.ur_run_uu, label='Upper Reservoir - Quick Runoff', width=2.0, edgecolor='none')
        ur_run_ul_ax.bar(bar_x, self.ur_run_ul, label='Upper Reservoir - Slow Runoff', width=1.0, edgecolor='none')
        ur_to_lr_run_ax.bar(bar_x, self.ur_to_lr_run, label='Upper Reservoir  - Percolation', width=1.0, edgecolor='none')
        l_res_sto_ax.plot(self.lr_sto_arr, lw=0.5, label='Lower Reservoir - Storage')

        snow_ax.fill_between(bar_x, 0, self.snow_arr, alpha=0.3)
        sm_ax.fill_between(bar_x, 0, self.sm_arr, alpha=0.3)
        u_res_sto_ax.fill_between(bar_x, 0, self.ur_sto_arr, alpha=0.3)
        l_res_sto_ax.fill_between(bar_x, 0, self.lr_sto_arr, alpha=0.3)

        text = np.array([
               'Output parameters file: %s' % os.path.basename(out_params_loc),
               'Output time series figure: %s' % os.path.basename(out_fig_loc),
               'Maximum input discharge = %0.4f' % self.q_act_arr_orig[self.off_idx:].max(),
               'Maximum simulated discharge = %0.4f' % self.q_sim_arr[self.off_idx:].max(),
               'Minimum input discharge = %0.4f' % self.q_act_arr_orig[self.off_idx:].min(),
               'Minimum simulated discharge = %0.4f' % self.q_sim_arr[self.off_idx:].min(),
               'Mean input discharge = %0.4f' % np.mean(self.q_act_arr_orig[self.off_idx:]),
               'Mean simulated discharge = %0.4f' % np.mean(self.q_sim_arr[self.off_idx:]),
               'Maximum input precipitation = %0.4f' % self.prec_arr_orig.max(),
               'Maximum snow depth = %0.4f' % self.snow_arr[self.off_idx:].max(),
               'Maximum liquid precipitation = %0.4f' % self.liqu_arr[self.off_idx:].max(),
               'Maximum input temperature = %0.4f' % self.temp_arr_orig[self.off_idx:].max(),
               'Minimum input temperature = %0.4f' % self.temp_arr_orig[self.off_idx:].min(),
               'Maximum potential evapotranspiration = %0.4f' % self.pet_arr_orig[self.off_idx:].max(),
               'Maximum simulated evapotranspiration = %0.4f' % self.evap_arr[self.off_idx:].max(),
               'Minimum potential evapotranspiration = %0.4f' % self.pet_arr_orig[self.off_idx:].min(),
               'Minimum simulated evapotranspiration = %0.4f' % self.evap_arr[self.off_idx:].min(),
               'Maximum simulated soil moisture = %0.4f' % self.sm_arr[self.off_idx:].max(),
               'Minimum simulated soil moisture = %0.4f' % self.sm_arr[self.off_idx:].min(),
               'Mean simulated soil moisture = %0.4f' % np.mean(self.sm_arr[self.off_idx:]),
               'Maximum total runoff = %0.4f' % self.tot_run_arr[self.off_idx:].max(),
               'Warm up steps = %d' % self.off_idx,
               'Note: X and Y axes\' units depend on inputs',
               'Nash-Sutcliffe = %0.4f' % self.ns,
               'Log Nash-Sutcliffe = %0.4f' % self.ln_ns,
               'Kling-Gupta = %0.4f' % self.kge,
               'Threshold temperature = %0.4f' % self.tt,
               'C$_{melt}$ = %0.4f' % self.c_melt,
               'Field capacity = %0.4f' % self.fc,
               'Beta = %0.4f' % self.beta,
               'Permanent wilting point = %0.4f' % self.pwp,
               'ur$_{thresh}$ = %0.4f' % self.ur_thresh,
               '$K_{uu}$ = %0.4f' % self.k_uu,
               '$K_{ul}$ = %0.4f' % self.k_ul,
               '$K_d$ = %0.4f' % self.k_d,
               '$K_{ll}$ = %0.4f' % self.k_ll,
               ])

        text = text.reshape(4, 9).T
        table = params_ax.table(cellText=text, loc='center', bbox=(0, 0, 1, 1), cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        params_ax.set_axis_off()

        plot_axes = [discharge_ax, prec_ax, temp_ax, pet_ax, snow_ax,
                     liqu_ax, sm_ax, tot_run_ax, u_res_sto_ax, ur_run_uu_ax,
                     ur_run_ul_ax, ur_to_lr_run_ax, l_res_sto_ax, vol_err_ax, balance_ax]
        for ax in plot_axes:
            ax.legend(framealpha=0.5)
            ax.grid()

        plt.savefig(out_fig_loc, dpi=self.fig_dpi, bbox='tight_layout')
        plt.close()

        if self.msgs:
            print 'Done saving output figure and parameters file.'

        if save_de:
            self.save_pop_opt(out_dir, out_suff)
        return

    def save_water_bal_opt(self, out_dir, out_suff, save_de=False):
        '''Save the output of the optimize function

        Just the water balance part
        '''
        assert self.simple_opt_flag or self.npop_opt_flag or self.kfold_opt_flag, \
               'Optimize first!'

        self._verify_init_model_01()
        self._reassign_arrs(0, self.n_recs_orig + 1)

        assert np.any(self.params_arr), 'HBV parameters are not defined!'
        assert self.q_act_arr_orig.shape[0] == self.q_sim_arr.shape[0], \
               'Original and simulated discharge have unequal steps!'

        if self.msgs:
            print '\nSaving water balance figure and parameters file...'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_fig_loc = os.path.join(out_dir, 'HBV_model_plot_%s.png' % out_suff)
        out_params_loc = os.path.join(out_dir, 'HBV_model_params_%s.csv' % out_suff)

        out_labs = []
        out_labs.extend(self.param_syms)
        out_labs.extend(['ns', 'ln_ns', 'kge', 'obj_ftn'])

        out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
        out_params_df['value'] = np.concatenate((self.params_arr, [self.ns, self.ln_ns, self.kge, self.obj_ftn_str]))
        out_params_df.to_csv(out_params_loc, sep=str(';'), index_label='param')


        cum_q_orig = np.cumsum(self.q_act_arr_orig[self.off_idx:]) / self.conv_ratio
        cum_q_sim = np.cumsum(self.comb_run_arr[self.off_idx:])
        vol_diff_arr = np.concatenate((np.full(shape=(self.off_idx - 1), fill_value=np.nan), cum_q_sim / cum_q_orig))

        bal_idxs = [0]
        bal_idxs.extend(range(self.off_idx, self.n_recs_orig, 365 * 2))

        act_bal_w_et_arr = []
        sim_bal_w_et_arr = []

        act_bal_wo_et_arr = []
        sim_bal_wo_et_arr = []

        for i in range(1, len(bal_idxs) - 1):
            # ET accounted for
            act_bal_w_et_arr.append((np.sum(self.q_act_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) / self.conv_ratio) / \
                                    (np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) - \
                                     np.sum(self.evap_arr[bal_idxs[i]:bal_idxs[i + 1]])))

            sim_bal_w_et_arr.append(np.sum(self.comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]]) / \
                                   (np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) - \
                                    np.sum(self.evap_arr[bal_idxs[i]:bal_idxs[i + 1]])))

            # ET not accounted for
            act_bal_wo_et_arr.append((np.sum(self.q_act_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]) / self.conv_ratio) / \
                                      np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]))

            sim_bal_wo_et_arr.append(np.sum(self.comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]]) / \
                                     np.sum(self.prec_arr_orig[bal_idxs[i]:bal_idxs[i + 1]]))

        act_bal_w_et_arr = np.concatenate(([np.nan, np.nan], act_bal_w_et_arr), axis=0)
        sim_bal_w_et_arr = np.concatenate(([np.nan, np.nan], sim_bal_w_et_arr), axis=0)

        act_bal_wo_et_arr = np.concatenate(([np.nan, np.nan], act_bal_wo_et_arr), axis=0)
        sim_bal_wo_et_arr = np.concatenate(([np.nan, np.nan], sim_bal_wo_et_arr), axis=0)

        plt.figure(figsize=(30, 15))
        t_rows = 6
        t_cols = 10

        plt.suptitle('HBV Flow Simulation - Water Balance\n')

        i = 0
        params_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        discharge_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        vol_err_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        balance_w_et_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1
        balance_wo_et_ax = plt.subplot2grid((t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
        i += 1

        discharge_ax.plot(self.q_act_arr_orig, 'r-', label='Actual', lw=0.8)
        discharge_ax.plot(self.q_sim_arr, 'b-', label='Simulated', lw=0.5, alpha=0.5)

        vol_err_ax.axhline(1.0, color='k')
        vol_err_ax.plot(vol_diff_arr, lw=0.5, label='Cumm. Runoff Error')
        vol_err_ax.set_ylim(0.5, 1.5)

        balance_w_et_ax.axhline(1.0, color='k')
        balance_w_et_ax.scatter(bal_idxs, act_bal_w_et_arr, c='r', label='Actual Volume (w ET)', alpha=0.6)
        balance_w_et_ax.scatter(bal_idxs, sim_bal_w_et_arr, c='b', label='Simulated Volume (w ET)', alpha=0.6)
        balance_w_et_ax.set_xlim(0, discharge_ax.get_xlim()[1])
        balance_w_et_ax.set_ylim(0.5, 1.5)

        balance_wo_et_ax.axhline(1.0, color='k')
        balance_wo_et_ax.scatter(bal_idxs, act_bal_wo_et_arr, c='r', label='Actual Volume (wo ET)', alpha=0.6)
        balance_wo_et_ax.scatter(bal_idxs, sim_bal_wo_et_arr, c='b', label='Simulated Volume (wo ET)', alpha=0.6)
        balance_wo_et_ax.set_xlim(0, discharge_ax.get_xlim()[1])
        balance_wo_et_ax.set_ylim(0.0, 1.2)

        text = np.array([
               'Output parameters file: %s' % os.path.basename(out_params_loc),
               'Output time series figure: %s' % os.path.basename(out_fig_loc),
               'Maximum input discharge = %0.4f' % self.q_act_arr_orig[self.off_idx:].max(),
               'Maximum simulated discharge = %0.4f' % self.q_sim_arr[self.off_idx:].max(),
               'Minimum input discharge = %0.4f' % self.q_act_arr_orig[self.off_idx:].min(),
               'Minimum simulated discharge = %0.4f' % self.q_sim_arr[self.off_idx:].min(),
               'Mean input discharge = %0.4f' % np.mean(self.q_act_arr_orig[self.off_idx:]),
               'Mean simulated discharge = %0.4f' % np.mean(self.q_sim_arr[self.off_idx:]),
               'Maximum input precipitation = %0.4f' % self.prec_arr_orig.max(),
               'Maximum snow depth = %0.4f' % self.snow_arr[self.off_idx:].max(),
               'Maximum liquid precipitation = %0.4f' % self.liqu_arr[self.off_idx:].max(),
               'Maximum input temperature = %0.4f' % self.temp_arr_orig[self.off_idx:].max(),
               'Minimum input temperature = %0.4f' % self.temp_arr_orig[self.off_idx:].min(),
               'Maximum potential evapotranspiration = %0.4f' % self.pet_arr_orig[self.off_idx:].max(),
               'Maximum simulated evapotranspiration = %0.4f' % self.evap_arr[self.off_idx:].max(),
               'Minimum potential evapotranspiration = %0.4f' % self.pet_arr_orig[self.off_idx:].min(),
               'Minimum simulated evapotranspiration = %0.4f' % self.evap_arr[self.off_idx:].min(),
               'Maximum simulated soil moisture = %0.4f' % self.sm_arr[self.off_idx:].max(),
               'Minimum simulated soil moisture = %0.4f' % self.sm_arr[self.off_idx:].min(),
               'Mean simulated soil moisture = %0.4f' % np.mean(self.sm_arr[self.off_idx:]),
               'Maximum total runoff = %0.4f' % self.tot_run_arr[self.off_idx:].max(),
               'Warm up steps = %d' % self.off_idx,
               'Note: X and Y axes\' units depend on inputs',
               'Nash-Sutcliffe = %0.4f' % self.ns,
               'Log Nash-Sutcliffe = %0.4f' % self.ln_ns,
               'Kling-Gupta = %0.4f' % self.kge,
               'Threshold temperature = %0.4f' % self.tt,
               'C$_{melt}$ = %0.4f' % self.c_melt,
               'Field capacity = %0.4f' % self.fc,
               'Beta = %0.4f' % self.beta,
               'Permanent wilting point = %0.4f' % self.pwp,
               'ur$_{thresh}$ = %0.4f' % self.ur_thresh,
               '$K_{uu}$ = %0.4f' % self.k_uu,
               '$K_{ul}$ = %0.4f' % self.k_ul,
               '$K_d$ = %0.4f' % self.k_d,
               '$K_{ll}$ = %0.4f' % self.k_ll,
               ])

        text = text.reshape(4, 9).T
        table = params_ax.table(cellText=text, loc='center', bbox=(0, 0, 1, 1), cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        params_ax.set_axis_off()

        plot_axes = [discharge_ax, vol_err_ax, balance_w_et_ax, balance_wo_et_ax]
        for ax in plot_axes:
            ax.legend(framealpha=0.5)
            ax.grid()

        plt.tight_layout(rect=[0, -0.1, 1, 0.95])
        plt.savefig(out_fig_loc, dpi=self.fig_dpi, bbox='tight_layout')
        plt.close()

        if self.msgs:
            print 'Done saving water balance figure and parameters file.'

        if save_de:
            self.save_pop_opt(out_dir, out_suff)
        return

    def save_kfolds_opt(self, out_dir, out_suff, save_hbv=False, save_de=False, save_wat_bal=False):
        '''Save the output of the optimize_kfolds function
        '''
        assert self.kfold_opt_flag, 'Call \'optimize_kfolds\' first!'

        if self.msgs:
            print '\nSaving kfolds output figure and parameters file...'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_fig_loc_1 = os.path.join(out_dir, 'HBV_kfolds_model_plots_%s.png' % out_suff)
        out_fig_loc_2 = os.path.join(out_dir, 'HBV_kfolds_best_params_%s.png' % out_suff)
        out_params_loc = os.path.join(out_dir, 'HBV_kfolds_model_params_%s.csv' % out_suff)

        self.params_kfolds_df.to_csv(out_params_loc, sep=str(';'))

        tot_plots = 1
        fig_rows = 10
        fig_cols = 5
        leg_rows = 1
        plot_rows = (tot_plots * fig_rows) + leg_rows  # leg_rows for legend
        plot_cols = 5 * 3

        plt.figure(figsize=(15, 9))

        title_str = 'HBV %d-fold calibration / validation results' % self.kfolds
        title_str += '\nTotal steps: %d, %d steps per fold' % (self.temp_arr_orig.shape[0], self.idxs_list[1])
        plt.suptitle(title_str, size=16)

        ns_ax = plt.subplot2grid((plot_rows, plot_cols), (0, 0), rowspan=fig_rows, colspan=fig_cols)
        ln_ns_ax = plt.subplot2grid((plot_rows, plot_cols), (0, fig_cols), rowspan=fig_rows, colspan=fig_cols)
        kge_ax = plt.subplot2grid((plot_rows, plot_cols), (0, fig_cols*2), rowspan=fig_rows, colspan=fig_cols)

        ps = ns_ax.matshow(self.ns_arr, vmin=0, vmax=1, cmap=cmaps.Blues, origin='lower')
        ln_ns_ax.matshow(self.ln_ns_arr, vmin=0, vmax=1, cmap=cmaps.Blues, origin='lower')
        kge_ax.matshow(self.kge_arr, vmin=0, vmax=1, cmap=cmaps.Blues, origin='lower')

        for s in zip(np.repeat(range(self.kfolds), self.kfolds), np.tile(range(self.kfolds), self.kfolds)):
            ns_ax.text(s[1], s[0], '%0.3f' % self.ns_arr[s[0], s[1]], va='center', ha='center')
            ln_ns_ax.text(s[1], s[0], '%0.3f' % self.ln_ns_arr[s[0], s[1]], va='center', ha='center')
            kge_ax.text(s[1], s[0], '%0.3f' % self.kge_arr[s[0], s[1]], va='center', ha='center')

        pos = 'bottom'

        axes_list = [ns_ax, ln_ns_ax, kge_ax]
        for ax_idx, ax in enumerate(axes_list):
            ax.xaxis.set_label_position(pos)
            ax.xaxis.set_ticks_position(pos)
            ax.set_xticks(range(0, self.kfolds))
            ax.set_xticklabels(range(1, self.kfolds + 1))
            if ax_idx != 1:
                ax.set_yticks(range(0, self.kfolds))
                ax.set_yticklabels(range(1, self.kfolds + 1))
                ax.set_ylabel('Periods')

        ns_ax.set_xlabel('NS')

        ln_ns_ax.set_xlabel('Ln_NS')
        plt.setp(ln_ns_ax.get_yticklabels(), visible=False)

        kge_ax.set_xlabel('KGE')
        kge_ax.yaxis.set_ticks_position('right')
        kge_ax.yaxis.set_label_position('right')

        cb_ax = plt.subplot2grid((plot_rows, plot_cols), (fig_rows, 0), rowspan=leg_rows, colspan=plot_cols)
        cb_ax.set_axis_off()
        cb = plt.colorbar(ps, ax=cb_ax, fraction=1.0, aspect=20, orientation='horizontal', extend='min')
        cb.set_ticks(np.arange(0, 1.01, 0.2))
        cb.set_label('Efficiency')
        plt.savefig(out_fig_loc_1, bbox='tight_layout', dpi=self.fig_dpi)

        self._plot_best(out_fig_loc_2)

        if self.msgs:
            print 'Done saving kfolds output figure and parameters file.'

        if save_hbv or save_wat_bal:
            save_de = True

        if save_de and save_hbv:
            self.save_pop_opt(out_dir, out_suff, save_hbv=save_hbv)

        if save_de and save_wat_bal:
            self.save_pop_opt(out_dir, out_suff, save_wat_bal=save_wat_bal)

        return

    def save_pop_opt(self, out_dir, out_suff, save_hbv=False, save_wat_bal=False):
        '''Save outputs of differential evolution
        '''
        assert self.simple_opt_flag or self.npop_opt_flag or self.kfold_opt_flag, \
               'Optimize first!'

        if self.msgs:
            print '\nSaving population tables and figures...'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.out_multi_pop_dir = out_dir
        self.out_multi_pop_suff = out_suff

        out_params_loc = os.path.join(out_dir, 'HBV_de_pop_model_best_params_%s.csv' % out_suff)
        if self.params_npops_df is not None:
            self.params_npops_df.to_csv(out_params_loc, sep=str(';'), index_label='pop_no')

        self.save_hbv = save_hbv
        self.save_wat_bal = save_wat_bal

        if (self.nprocs == 1) or self.debug_mode:
            corrs_arr = []
            for j, opt_res in enumerate(self.opt_multi_pop_res):
                corrs_arr.append(self._plot_de((j, opt_res)))
        else:
            corrs_arr = self.de_pool.map(self._plot_de, enumerate(self.opt_multi_pop_res))
            self.de_pool.clear()

        corrs_arr = [np.array(corr_arr) for corr_arr in corrs_arr]

        for j in range(self.npops):
            plt.figure(figsize=(7, 7))
            plt.title('HBV parameter correlation matrix\nfor population no: %0.3d' % j)

#            corrs_arr = np.array(corrs_arr)
            ranks = pd.DataFrame(corrs_arr[j])
            corr_arr = ranks.rank().values
            corr_arr = fill_correl_mat(corr_arr)
#            corr_arr = fill_correl_mat(corrs_arr[j])

            corrs_ax = plt.subplot(111)
            corrs_ax.matshow(corr_arr, vmin=0, vmax=1, cmap=cmaps.Blues, origin='lower')
            for s in zip(np.repeat(range(self.n_params), self.n_params), np.tile(range(self.n_params), self.n_params)):
                corrs_ax.text(s[1], s[0], '%0.2f' % (corr_arr[s[0], s[1]]), va='center', ha='center',
                              fontsize=self.tick_font_size)

            corrs_ax.set_xticks(range(0, self.n_params))
            corrs_ax.set_xticklabels(self.param_syms)
            corrs_ax.set_yticks(range(0, self.n_params))
            corrs_ax.set_yticklabels(self.param_syms)

            corrs_ax.spines['left'].set_position(('outward', 10))
            corrs_ax.spines['right'].set_position(('outward', 10))
            corrs_ax.spines['top'].set_position(('outward', 10))
            corrs_ax.spines['bottom'].set_position(('outward', 10))

            corrs_ax.tick_params(labelleft=True,
                                 labelbottom=True,
                                 labeltop=False,
                                 labelright=True)

            plt.setp(corrs_ax.get_xticklabels(), size=self.tick_font_size, rotation=45)
            plt.setp(corrs_ax.get_yticklabels(), size=self.tick_font_size)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.90)

            plt.savefig(os.path.join(out_dir,
                                     'params_corrs_%s_pop_no_%0.3d.png' % (out_suff, j)),
                        dpi=self.fig_dpi, bbox='tight_layout')
            plt.close()

        if self.msgs:
            print 'Done saving population figures.'

        if self.nprocs > 1:
            for key in locals().keys():
                exec('del key')
        return

    def _run_sim(self):
        '''Run a simulation based on given data inside the model
        '''
        all_output = loop_HBV_py(
                     self.n_recs,
                     self.conv_ratio,
                     self.params_arr,
                     self.ini_arr,
                     self.temp_arr,
                     self.prec_arr,
                     self.pet_arr)

        self.snow_arr = all_output[:, 0]
        self.liqu_arr = all_output[:, 1]
        self.sm_arr = all_output[:, 2]
        self.tot_run_arr = all_output[:, 3]
        self.evap_arr = all_output[:, 4]
        self.comb_run_arr = all_output[:, 5]
        self.q_sim_arr = all_output[:, 6]
        self.ur_sto_arr = all_output[:, 7]
        self.ur_run_uu = all_output[:, 8]
        self.ur_run_ul = all_output[:, 9]
        self.ur_to_lr_run = all_output[:, 10]
        self.lr_sto_arr = all_output[:, 11]
        self.lr_run_arr = all_output[:, 12]
        return

#    def _run_sim_c(self):
#        '''Run a simulation based on given data inside the model
#        '''
#        all_output = loop_HBV_cpy(
#                     self.n_recs,
#                     self.conv_ratio,
#                     self.params_arr,
#                     self.ini_arr,
#                     self.temp_arr,
#                     self.prec_arr,
#                     self.pet_arr)
#
#        self.snow_arr = all_output[:, 0]
#        self.liqu_arr = all_output[:, 1]
#        self.sm_arr = all_output[:, 2]
#        self.tot_run_arr = all_output[:, 3]
#        self.evap_arr = all_output[:, 4]
#        self.comb_run_arr = all_output[:, 5]
#        self.q_sim_arr = all_output[:, 6]
#        self.ur_sto_arr = all_output[:, 7]
#        self.ur_run_uu = all_output[:, 8]
#        self.ur_run_ul = all_output[:, 9]
#        self.ur_to_lr_run = all_output[:, 10]
#        self.lr_sto_arr = all_output[:, 11]
#        self.lr_run_arr = all_output[:, 12]
#        return

    def _get_kfold_opt_mp(self, n):
        self._reassign_arrs(self.idxs_list[n], self.idxs_list[n + 1])
        return self._get_opt_mp(n)

    def _get_opt_mp(self, i):
        return hbv_de(
               self.bounds_arr,
               self.n_recs,
               self.off_idx,
               self.conv_ratio,
               self.ini_arr,
               self.temp_arr,
               self.prec_arr,
               self.pet_arr,
               self.q_act_arr,
               self.obj_ftn_flags_arr,
               self.obj_ftn_wts_arr,
               self.mu_sc_fac,
               self.cr_cnst,
               self.pop_size,
               self.de_iter_max,
               self.de_max_unsc_iters,
               self.de_tol)

    def _plot_de(self, j_opt_res):
        xx, yy = np.meshgrid(np.arange(-0.5, self.n_params), np.arange(-0.5, self.n_stats_cols))

        plt.figure(figsize=(13, 7))

        j, opt_res = j_opt_res
        pop = opt_res['pop']

        if self.save_hbv or self.save_water_bal_opt:
            self._reassign_arrs(0, self.n_recs_orig + 1)
            old_msgs_flag = self.msgs
            self.msgs = False
            self.tt, \
            self.c_melt, \
            self.fc, \
            self.beta, \
            self.pwp, \
            self.ur_thresh, \
            self.k_uu, \
            self.k_ul, \
            self.k_d, \
            self.k_ll = opt_res['params']
            self.params_arr = opt_res['params']

            self._run_sim()
            if self.save_hbv:
                self.save_simple_opt(self.out_multi_pop_dir,
                                     'pop_%0.3d_%s' % (j, self.out_multi_pop_suff),
                                     save_de=False)
            elif self.save_wat_bal:
                self.save_water_bal_opt(self.out_multi_pop_dir,
                                    'pop_%0.3d_%s' % (j, self.out_multi_pop_suff),
                                    save_de=False)
            self.msgs = old_msgs_flag

        curr_min = pop.min(axis=0)
        curr_max = pop.max(axis=0)
        curr_mean = pop.mean(axis=0)
        curr_stdev = pop.std(axis=0)
        min_opt_bounds = self.bounds_arr.min(axis=1)
        max_opt_bounds = self.bounds_arr.max(axis=1)

        stats_arr = np.vstack([curr_min, curr_max, curr_mean, curr_stdev, min_opt_bounds, max_opt_bounds])

        stats_ax = plt.subplot2grid((self.n_rows, self.n_cols), (0, 0), rowspan=1, colspan=self.n_cols)
        stats_ax.pcolormesh(xx, yy, stats_arr, cmap=cmaps.Blues,
                            vmin=-np.inf, vmax=np.inf)

        for x in xrange(self.n_stats_cols):
            for y in xrange(self.n_params):
                stats_ax.text(y, x, ('%3.4f' % stats_arr[x, y]).rstrip('0'),
                              va='center', ha='center', fontsize=self.tick_font_size)

        stats_ax.set_xticks(range(0, self.n_params))
        stats_ax.set_xticklabels(self.param_syms)
        stats_ax.set_xlim(-0.5, self.n_params - 0.5)
        stats_ax.set_yticks(range(0, self.n_stats_cols))
        stats_ax.set_ylim(-0.5, self.n_stats_cols - 0.5)
        stats_ax.set_yticklabels(self.stats_cols)

        stats_ax.spines['left'].set_position(('outward', 10))
        stats_ax.spines['right'].set_position(('outward', 10))
        stats_ax.spines['top'].set_position(('outward', 10))
        stats_ax.spines['bottom'].set_visible(False)

        stats_ax.set_ylabel('Statistics', size=self.tick_font_size)

        stats_ax.tick_params(labelleft=True,
                             labelbottom=False,
                             labeltop=True,
                             labelright=True)

        stats_ax.xaxis.set_ticks_position('top')

        norm_pop = (pop - self.bounds_arr[:, 0]) / (self.bounds_arr[:, 1] - self.bounds_arr[:, 0])

        params_ax = plt.subplot2grid((self.n_rows, self.n_cols), (1, 0), rowspan=2, colspan=self.n_cols, sharex=stats_ax)
        plot_range = range(0, self.bounds_arr.shape[0])
        for i in range(norm_pop.shape[0]):
            params_ax.plot(plot_range, norm_pop[i], alpha=0.1)

        params_ax.set_ylim(0., 1.)
        params_ax.set_xticks(range(pop.shape[1]))
        params_ax.set_xticklabels(self.param_syms)
        params_ax.set_xlim(-0.5, self.n_params - 0.5)
        params_ax.set_ylabel('Normalized value', size=self.tick_font_size)

        title_str = 'Normalized HBV parameters from DE\n'
        title_str += 'Population number: %d of %d, Number of individuals: %d, ' % (j, self.npops - 1, pop.shape[0])
        title_str += 'Final generation: %d, ' % opt_res['n_gens']
        title_str += 'Max. generations: %d, ' % self.de_iter_max
        title_str += '\nTotal calls: %d, ' % opt_res['n_calls']
        title_str += 'Successful attempts: %d, ' % opt_res['n_succ']
        title_str += 'Last successful attempt at: %d, ' % opt_res['lst_succ_try']
        title_str += '\nContinuous iterations without success: %d, ' % opt_res['cont_iter']
        title_str += 'Final tolerance: %0.2e, ' % opt_res['fin_tol']
        title_str += '\n[%0.2f <= F < %0.2f], ' % (self.mu_sc_fac[0], self.mu_sc_fac[1])
        title_str += '[%0.2f <= C < %0.2f], ' % (self.cr_cnst[0], self.cr_cnst[1])
        title_str += 'tol_limit: %0.2e' % self.de_tol


        de_mu_sc_arr_tot = np.array(opt_res['total_vars'])[:, 0]
        de_cr_cnst_arr_tot = np.array(opt_res['total_vars'])[:, 1]

        de_mu_sc_arr = np.array(opt_res['accept_vars'])[:, 0]
        de_cr_cnst_arr = np.array(opt_res['accept_vars'])[:, 1]

        de_mu_sc_arr_tot_hist, _ = np.histogram(de_mu_sc_arr_tot, bins=self.hist_bins, range=(self.mu_sc_fac[0], self.mu_sc_fac[1]))
        de_mu_sc_arr_hist, mu_sc_edges = np.histogram(de_mu_sc_arr, bins=self.hist_bins, range=(self.mu_sc_fac[0], self.mu_sc_fac[1]))
        de_mu_sc_arr_hist = np.array(de_mu_sc_arr_hist, dtype='float64') / de_mu_sc_arr_tot_hist
        mu_sc_edges = [(mu_sc_edges[i] + mu_sc_edges[i + 1]) * 0.5 for i in range(mu_sc_edges.shape[0] - 1)]

        de_cr_cnst_arr_tot_hist, _ = np.histogram(de_cr_cnst_arr_tot, bins=self.hist_bins, range=(self.cr_cnst[0], self.cr_cnst[1]))
        de_cr_cnst_arr_hist, cr_cnst_edges = np.histogram(de_cr_cnst_arr, bins=self.hist_bins, range=(self.cr_cnst[0], self.cr_cnst[1]))
        de_cr_cnst_arr_hist = np.array(de_cr_cnst_arr_hist, dtype='float64') / de_cr_cnst_arr_tot_hist
        cr_cnst_edges = [(cr_cnst_edges[i] + cr_cnst_edges[i + 1]) * 0.5 for i in range(cr_cnst_edges.shape[0] - 1)]


        de_mu_sc_ax = plt.subplot2grid((self.n_rows, self.n_cols), (3, 0), rowspan=1, colspan=1)
        de_mu_sc_ax.plot(mu_sc_edges, de_mu_sc_arr_hist, alpha=0.5, marker='o', ms=4)
        de_mu_sc_ax.set_ylabel('Relative acceptence', size=self.tick_font_size)
        de_mu_sc_ax.set_ylim(0, 1)

        de_mu_sc_ax.tick_params(labelleft=True,
                                labelbottom=False,
                                labeltop=False,
                                labelright=False)
        de_mu_sc_ax.set_xlim(self.mu_sc_fac[0], self.mu_sc_fac[1])

        de_cr_cnst_ax = plt.subplot2grid((self.n_rows, self.n_cols), (3, 1), rowspan=1, colspan=1)
        de_cr_cnst_ax.plot(cr_cnst_edges, de_cr_cnst_arr_hist, alpha=0.5, marker='o', ms=4)
        de_cr_cnst_ax.set_ylim(0, 1)
        de_cr_cnst_ax.tick_params(labelleft=False,
                                  labelbottom=False,
                                  labeltop=False,
                                  labelright=True)
        de_cr_cnst_ax.set_ylabel('Relative acceptence', size=self.tick_font_size)
        de_cr_cnst_ax.yaxis.set_label_position("right")
        de_cr_cnst_ax.set_xlim(self.cr_cnst[0], self.cr_cnst[1])

        de_mu_sc_ax_tot = plt.subplot2grid((self.n_rows, self.n_cols), (4, 0), rowspan=1, colspan=1, sharex=de_mu_sc_ax)
        de_mu_sc_ax_tot.hist(de_mu_sc_arr_tot, self.hist_bins, normed=False, facecolor='blue', alpha=0.25, range=(self.mu_sc_fac[0], self.mu_sc_fac[1]))
        de_mu_sc_ax_tot.set_xlabel('Mutation scale factor (F)', size=self.tick_font_size)
        de_mu_sc_ax_tot.set_ylabel('Total count', size=self.tick_font_size)
        de_mu_sc_ax_tot.set_xlim(self.mu_sc_fac[0], self.mu_sc_fac[1])

        de_cr_cnst_ax_tot = plt.subplot2grid((self.n_rows, self.n_cols), (4, 1), rowspan=1, colspan=1, sharex=de_cr_cnst_ax)
        de_cr_cnst_ax_tot.hist(de_cr_cnst_arr_tot, self.hist_bins, normed=False, facecolor='blue', alpha=0.25, range=(self.cr_cnst[0], self.cr_cnst[1]))
        de_cr_cnst_ax_tot.set_xlabel('Recombination factor (C)', size=self.tick_font_size)
        de_cr_cnst_ax_tot.tick_params(labelleft=False,
                                      labelbottom=True,
                                      labeltop=False,
                                      labelright=True)

        de_cr_cnst_ax_tot.set_ylabel('Total count', size=self.tick_font_size)
        de_cr_cnst_ax_tot.yaxis.set_label_position("right")
        de_cr_cnst_ax_tot.set_xlim(self.cr_cnst[0], self.cr_cnst[1])

        ax_list = [stats_ax, params_ax, de_mu_sc_ax, de_cr_cnst_ax, de_mu_sc_ax_tot, de_cr_cnst_ax_tot]
        for ax_idx, ax in enumerate(ax_list):
            plt.setp(ax.get_xticklabels(), size=self.tick_font_size)
            plt.setp(ax.get_yticklabels(), size=self.tick_font_size)
            if ax_idx > 0:
                ax.grid()
            if ax_idx > 1:
                ax.locator_params(axis='y', nbins=5)

        plt.suptitle(title_str, size=self.tick_font_size + 3)
        plt.tight_layout(rect=[0, 0, 1, 0.87])
        plt.subplots_adjust(hspace=0.3)
        out_stats_path = os.path.join(self.out_multi_pop_dir, 'pop_overview_%0.3d_%s.png' % (j, self.out_multi_pop_suff))
        plt.savefig(out_stats_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(13, 7))
        scatter_range = np.arange(de_mu_sc_arr.shape[0])

        mu_evo_ax = plt.subplot2grid((self.n_rows, self.n_cols), (0, 0), rowspan=1, colspan=self.n_cols)
        mu_evo_ax.scatter(scatter_range, de_mu_sc_arr, alpha=0.2, s=1.0)
        mu_evo_ax.set_ylim(self.mu_sc_fac[0], self.mu_sc_fac[1])
        mu_evo_ax.set_ylabel('F', size=self.tick_font_size)
        mu_evo_ax.tick_params(labelbottom=False, labeltop=True)
        mu_evo_ax.set_xlabel('Success iteration number of F, C, and Log objective', size=self.tick_font_size)
        mu_evo_ax.xaxis.set_ticks_position('top')
        mu_evo_ax.xaxis.set_label_position('top')

        cr_evo_ax = plt.subplot2grid((self.n_rows, self.n_cols), (1, 0), rowspan=1, colspan=self.n_cols, sharex=mu_evo_ax)
        cr_evo_ax.scatter(scatter_range, de_cr_cnst_arr, alpha=0.2, s=1.0)
        cr_evo_ax.set_ylim(self.cr_cnst[0], self.cr_cnst[1])
        cr_evo_ax.set_ylabel('C', size=self.tick_font_size)
        cr_evo_ax.tick_params(labelbottom=False)

        de_obj_evo_arr = np.log(np.array(opt_res['accept_vars'])[:, 2])

        obj_evo_ax = plt.subplot2grid((self.n_rows, self.n_cols), (2, 0), rowspan=1, colspan=self.n_cols, sharex=mu_evo_ax)
        obj_evo_ax.scatter(scatter_range, de_obj_evo_arr, alpha=0.2, s=1.0)
        obj_evo_ax.set_ylim(de_obj_evo_arr.min() - 1, de_obj_evo_arr[self.pop_size:self.pop_size*2].max())
        obj_evo_ax.set_ylabel('Log objective', size=self.tick_font_size)
        obj_evo_ax.set_xlim(0, obj_evo_ax.get_xlim()[1])
        obj_evo_ax.tick_params(labelbottom=False)

        acc_evo_arr, acc_evo_cts = np.unique(np.array(opt_res['accept_vars'])[:, 3], return_counts=True)

        acc_evo_ax = plt.subplot2grid((self.n_rows, self.n_cols), (3, 0), rowspan=1, colspan=self.n_cols)
        acc_evo_ax.scatter(acc_evo_arr, acc_evo_cts, alpha=0.5, s=1.5)
        acc_evo_ax.set_ylabel('Accept counts', size=self.tick_font_size)
        acc_evo_ax.set_xlim(0, acc_evo_ax.get_xlim()[1])
        acc_evo_ax.set_xlabel('Generation', size=self.tick_font_size)

        pop_obj_ax = plt.subplot2grid((self.n_rows, self.n_cols), (4, 0), rowspan=1, colspan=self.n_cols)
        pop_obj_ax.scatter(range(self.pop_size), opt_res['pop_obj_vals'], alpha=0.9, s=1.5)
        pop_obj_ax.set_ylabel('Objective', size=self.tick_font_size)
        pop_obj_ax.set_xlim(0, pop_obj_ax.get_xlim()[1])
        pop_obj_ax.set_xlabel('Final individual', size=self.tick_font_size)

        ax_list = [mu_evo_ax, cr_evo_ax, obj_evo_ax, acc_evo_ax, pop_obj_ax]
        for ax in ax_list:
            plt.setp(ax.get_xticklabels(), size=self.tick_font_size)
            plt.setp(ax.get_yticklabels(), size=self.tick_font_size)
            ax.grid()
            ax.locator_params(axis='y', nbins=5)

        plt.suptitle('DE parameter evolution, Population number: %d of %d' % (j, self.npops - 1), size=self.tick_font_size + 3)
        plt.tight_layout(rect=[0, 0, 1.0, 0.97])
        plt.subplots_adjust(hspace=0.4)
        out_evo_path = os.path.join(self.out_multi_pop_dir, 'de_param_evo_pop_%0.3d_%s.png' % (j, self.out_multi_pop_suff))
        plt.savefig(out_evo_path, dpi=self.fig_dpi, bbox_inches='tight')
        plt.close()

        for key in locals().keys():
            if (key != 'pop') and (not '__' in key):
                exec('del key')
        return pop

    def _chk_nans_in_pop(self):
        if not self.opt_multi_pop_res:
            raise AssertionError('No population to check for invalid values!')

        for opt_res in self.opt_multi_pop_res:
            if np.any(np.logical_not(opt_res['params'])) or \
               np.any(np.logical_not(opt_res['pop'])) or \
               np.any(np.logical_not(opt_res['pop_obj_vals'])):
                   raise AssertionError('Invalid values in optimization results!')
        return

    def _plot_best(self, out_fig_loc):
        plt.figure(figsize=(15, 9))
        best_params_arr = []
        for opt_res in self.opt_multi_pop_res:
            best_params_arr.append(opt_res['params'])
        best_params_arr = np.array(best_params_arr)

        curr_min = best_params_arr.min(axis=0)
        curr_max = best_params_arr.max(axis=0)
        curr_mean = best_params_arr.mean(axis=0)
        curr_stdev = best_params_arr.std(axis=0)
        min_opt_bounds = self.bounds_arr.min(axis=1)
        max_opt_bounds = self.bounds_arr.max(axis=1)
        xx, yy = np.meshgrid(np.arange(-0.5, self.n_params), np.arange(-0.5, self.n_stats_cols))

        stats_arr = np.vstack([curr_min, curr_max, curr_mean, curr_stdev, min_opt_bounds, max_opt_bounds])

        stats_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
        stats_ax.pcolormesh(xx, yy, stats_arr, cmap=cmaps.Blues,
                            vmin=-np.inf, vmax=np.inf)

        for x in xrange(self.n_stats_cols):
            for y in xrange(self.n_params):
                stats_ax.text(y, x, ('%3.4f' % stats_arr[x, y]).rstrip('0'),
                              va='center', ha='center', fontsize=self.tick_font_size + 3)

        stats_ax.set_xticks(range(0, self.n_params))
        stats_ax.set_xticklabels(self.param_syms)
        stats_ax.set_xlim(-0.5, self.n_params - 0.5)
        stats_ax.set_yticks(range(0, self.n_stats_cols))
        stats_ax.set_ylim(-0.5, self.n_stats_cols - 0.5)
        stats_ax.set_yticklabels(self.stats_cols)

        stats_ax.spines['left'].set_position(('outward', 10))
        stats_ax.spines['right'].set_position(('outward', 10))
        stats_ax.spines['top'].set_position(('outward', 10))
        stats_ax.spines['bottom'].set_visible(False)

        stats_ax.set_ylabel('Statistics')

        stats_ax.tick_params(labelleft=True,
                             labelbottom=False,
                             labeltop=True,
                             labelright=True)

        stats_ax.xaxis.set_ticks_position('top')

        norm_pop = (best_params_arr - self.bounds_arr[:, 0]) / (self.bounds_arr[:, 1] - self.bounds_arr[:, 0])

        params_ax = plt.subplot2grid((4, 1), (1, 0), rowspan=3, colspan=1, sharex=stats_ax)
        plot_range = range(0, self.bounds_arr.shape[0])
        plt_texts = []
        for i in range(self.kfolds):
            params_ax.plot(plot_range, norm_pop[i], alpha=0.85, label='Fold no: %d' % i)
            for j in range(self.bounds_arr.shape[0]):
                plt_texts.append(params_ax.text(plot_range[j], norm_pop[i, j], ('%3.4f' % best_params_arr[i, j]).rstrip('0'),
                                                va='top', ha='left', fontsize=self.tick_font_size))

        adjust_text(plt_texts, only_move={'points':'y', 'text':'y'})

        params_ax.set_ylim(0., 1.)
        params_ax.set_xticks(range(best_params_arr.shape[1]))
        params_ax.set_xticklabels(self.param_syms)
        params_ax.set_xlim(-0.5, self.n_params - 0.5)
        params_ax.set_ylabel('Normalized value')
        params_ax.grid()
        params_ax.legend(framealpha=0.5)

        title_str = 'Comparison of best parameters'
        plt.suptitle(title_str, size=self.tick_font_size + 10)
        plt.subplots_adjust(hspace=0.15)
        plt.savefig(out_fig_loc, bbox='tight_layout', dpi=self.fig_dpi)
        return

    def sim_rnd_sims(self):
        self._reset_params()
        self._verify_init_model_01()
        self._verify_init_model_02()
        self._verify_de_params()

        n_rand_params = 50
        rand_params_arr = np.full((self.bounds_arr.shape[0], n_rand_params), np.nan)

        obj_ftn_prm_arr = np.full((n_rand_params, 3), np.nan)
        obj_ftn_q_arr = np.full((n_rand_params, 3), np.nan)

        for i in range(self.bounds_arr.shape[0]):

            params_strt_val = self.bounds_arr[i, 0] + \
                              (self.bounds_arr[i, 1] - self.bounds_arr[i, 0])*np.random.random()

            params_end_val = self.bounds_arr[i, 0] + \
                             (self.bounds_arr[i, 1] - self.bounds_arr[i, 0])*np.random.random()

            if params_strt_val > params_end_val:
                params_end_val, params_strt_val = params_strt_val, params_end_val

            print 'Param_%d:' % i, params_strt_val, params_end_val
            rand_params_arr[i, :] = np.linspace(params_strt_val, params_end_val, n_rand_params)

        for i in range(n_rand_params):
            self.params_arr = rand_params_arr[:, i]
            print '\n%0.3d - parameters are:\n' % i, self.params_arr
            self.simple_opt_flag = False
            self._run_sim()
            self.simple_opt_flag = True
            self.tt, \
            self.c_melt, \
            self.fc, \
            self.beta, \
            self.pwp, \
            self.ur_thresh, \
            self.k_uu, \
            self.k_ul, \
            self.k_d, \
            self.k_ll = self.params_arr
#            self.save_simple_opt(os.path.join(os.getcwd(), 'q_err_test_auto_18'), 'q_err_auto_%0.3d' % i)
            obj_ftn_prm_arr[i, :] = self.ns, self.ln_ns, self.kge
            print '%0.3d - prm_efficiences are:\n' % i, obj_ftn_prm_arr[i, :]

        self.params_arr = rand_params_arr[:, 0]
        self.simple_opt_flag = False
        self._run_sim()
        self.simple_opt_flag = True
        q_1_ser = self.q_sim_arr.copy()

        self.params_arr = rand_params_arr[:, -1]
        self.simple_opt_flag = False
        self._run_sim()
        self.simple_opt_flag = True
        q_2_ser = self.q_sim_arr.copy()

        alphas_rng = np.linspace(1, 0, n_rand_params)

        for i in range(n_rand_params):
            self.q_sim_arr = (alphas_rng[i] * q_1_ser) + ((1 - alphas_rng[i]) * q_2_ser)
            obj_ftn_q_arr[i, :] = self.ns, self.ln_ns, self.kge
            print '%0.3d - q_efficiences are:\n' % i, obj_ftn_q_arr[i, :]

        plt.figure(figsize=(10, 6), dpi=150)
        lin_rng = range(0, n_rand_params)
        plt.scatter(lin_rng, obj_ftn_prm_arr[:, 0], label='NS_prm', marker='o')
#        plt.scatter(lin_rng, obj_ftn_prm_arr[:, 1], label='Ln_NS_prm', m='o')
        plt.scatter(lin_rng, obj_ftn_prm_arr[:, 2], label='KGE_prm', marker='o')

        plt.scatter(lin_rng, obj_ftn_q_arr[:, 0], label='NS_q', marker='+')
#        plt.scatter(lin_rng, obj_ftn_q_arr[:, 1], label='Ln_NS_q', m='+')
        plt.scatter(lin_rng, obj_ftn_q_arr[:, 2], label='KGE_q', marker='+')

        plt.legend(framealpha=0.5)
        plt.grid()
        plt.xlabel('Iteration number')
        plt.ylabel('Efficiency')
        plt.title('HBV simulations for random parameters')
        plt.savefig(os.path.join(os.getcwd(), 'q_err_test_auto_17', 'efficiencies_17_19.png'))
        plt.close()
        return


if __name__ == '__main__':
    pass