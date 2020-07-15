'''
The Normal copula infilling

Faizan Anwar, IWS
'''

from __future__ import unicode_literals
import timeit
from pickle import dump
from sys import exc_info
from random import random
from os import mkdir as os_mkdir
from math import log as mlog, ceil
from itertools import combinations
from traceback import format_exception
from os.path import join as os_join, exists as os_exists

from pandas import (read_csv,
                    to_datetime,
                    DataFrame,
                    Series,
                    date_range,
                    to_numeric,
                    read_pickle,
                    Index)
from numpy import (intersect1d,
                   vectorize,
                   logical_not,
                   isnan,
                   where,
                   linspace,
                   logical_or,
                   full,
                   array,
                   isclose,
                   linalg,
                   matmul,
                   any as np_any,
                   ediff1d,
                   fabs,
                   round as np_round,
                   repeat,
                   tile,
                   nan,
                   seterr,
                   abs as np_abs,
                   logical_and,
                   isfinite,
                   inf,
                   all as np_all,
                   append,
                   delete,
                   set_printoptions,
                   mgrid,
                   nanmax,
                   unique,
                   argsort,
                   sum as np_sum)

import matplotlib.cm as cmaps
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from adjustText import adjust_text
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathos.multiprocessing import ProcessPool as mp_pool

import pyximport
pyximport.install()

from xorshift_cy import gen_n_rns_arr
from normcop_cyftns import (get_corrcoeff,
                            get_dist,
                            fill_correl_mat,
                            norm_cdf_py,
                            norm_pdf_py,
                            norm_ppf_py,
                            get_kge_py,
                            norm_ppf_py_arr,
                            bi_var_copula,
                            get_ns_py,
                            bivar_gau_cop_arr,
                            tau_sample,
                            get_asymms_sample,
                            get_ln_ns_py)

plt.ioff()
set_printoptions(precision=6,
                 threshold=2000,
                 linewidth=200000,
                 formatter={'float': '{: 0.6f}'.format})

seterr(all='raise')

__all__ = ['NormCopulaInfill']


class NormCopulaInfill:
    '''
    Implementation of Bardossy & Pegram 2014 with some bells and jingles.

    Description
    -----------

    To infill missing time series data of a given station(s) using
    neighboring stations and the multivariate normal copula.
    Can be used for infilling time series data that acts like stream
    discharge or precipitation. After initiation, calling the infill method
    will start infilling based on the criteria explained below.

    Parameters
    ----------
    in_var_file: unicode, DataFrame_like
        Location of the file that holds the input time series data.
        The file should have its first column as time. The header
        should be the names of the stations. Any valid separator is
        allowed.
    in_coords_file: unicode, DataFrame_like
        Location of the file that has stations' coordinates.
        The names of the stations should be the first column. The header
        should have the \'X\' for eastings, \'Y\' for northings. Rest
        is ignored. Any valid separator is allowed.
    out_dir: unicode
        Location of the output directory. Will be created if it does
        not exist. All the ouputs are stored inside this directory.
    infill_stns: list_like
        Names of the stations that should be infilled. These names
        should be in the in_var_file header and the index of
        in_coords_file.
    min_valid_vals: integer
        The minimum number of the union of one station with others
        so that all have valid values. This is different in different
        cases. e.g. for calculating the long term correlation only
        two stations are used but during infilling all stations
        should satisfy this condition with respect to each other. i.e.
        only those days are used to make the correlation matrix on
        which all the neighboring stations have valid
        values at every step.
    infill_interval_type: unicode
        A string that is either: \'slice\', \'indiv\', \'all\'.
        slice: all steps in between infill_dates_list are
        considered for infilling.
        indiv: only fill on these dates.
        all: fill wherever there is a missing value.
    infill_dates_list: list_like, date_like
        A list containing the dates on which infilling is done.
        The way this list is used depends on the value of
        infill_interval_type.
        if infill_interval_type is \'slice\' and infill_dates_list
        is \'[date_1, date_2]\' then all steps in between and
        including date_1 and date_2 are infilled provided that
        neighboring stations have enough valid values at every
        step.
    n_nrst_stns_min: integer
        Number of minimum neighbors to use for infilling.
        Default is 1.
    n_nrst_stns_max: integer
        Number of maximum stations to use for infilling.
        Normally n_nrst_stns_min stations are used but it could be
        that at a given step one of the neighbors is missing a
        value in that case we have enough stations to choose from.
        This number depends on the given dataset. It cannot be
        less than n_nrst_stns_min.
        Default is 1.
    ncpus: integer
        Number of processes to initiate in case of methods that allow for
        multiprocessing. Ideally equal to the number of cores available.
        Default is 1.
    skip_stns: list_like
        The names of the stations that should not be used while
        infilling. Normally, after calling the
        \'cmpt_plot_rank_corr_stns\', one can see the stations
        that do not correlate well with the infill_stns.
        Default is None.
    sep: unicode
        The type of separator used in in_var_file and in_coords_file.
        Both of them should be similar. The output will have this as
        the separator as well.
        Default is \';\'.
    time_fmt: unicode
        Format of the time in the in_var_file. This is required to
        convert the string time to a datetime object that allows
        for a more accurate indexing. Any valid time format from the datetime
        module can be used.
        Default is \'%Y-%m-%d\'.
    freq: unicode or pandas datetime offset
        The type of interval used in the in_var_file. e.g. for
        days it is \'D\'. It is better to take a look at the
        pandas datetime offsets.
        Default is \'D\'.
    verbose: bool
        If True, print activity messages.
        Default is True.

    Post Initiation Parameters
    --------------------------
    Several attributes can be changed after initaiting the NormCopulaInfill
    object. These should be changed before calling any of the methods.

    nrst_stns_type: unicode
        Criteria to select neighbors. Can be \'rank\' (rank correlation),
        \'dist\' (2D proximity) or \'symm\' (rank correlation and symmetry).
        If infill_type is \'precipitation\' then nrst_stns_type cannot be
        \'symm\'. Default is \'symm\' if infill_type is
        \'discharge\' and \'rank\' if infill_type is \'precipitation\'.
    n_discrete: integer
        The number of intervals used to discretize the range of values
        while calculating the conditional distributions. The more the
        better but computation time will increase.
        Default is 300.
    n_norm_symm_flds: integer
        If nrst_stns_type is \'symm\', n_norm_symm_flds is the number of
        random bivariate normal copulas for a given correlation that are
        used to establish maximum and minimum asymmetry possible. If the
        asymmetry of the sample is within these bounds then it is taken
        as a neighbor and further used to infill. The order of these
        neighbors depend on the strength of their absolute correlation.
        Default is 100.
    fig_size_long: tuple of floats or integers
        Size of the figures (width, height) in inches. Default is
        (20, 7). This does not apply to diagnostic plots.
    out_fig_dpi: integer
        Dot-per-inch of the output figures. Default is 150.
    out_fig_fmt: unicode
        Output figure format. Any format supported by matplotlib.
        Default is png.
    conf_heads: list of unicodes
        Labels of the confidence intervals that appear in the output confidence
        values dataframes and figures.
        Default is
        ['var_0.05', 'var_0.25', 'var_0.50', 'var_0.75', 'var_0.95'].
    conf_probs: list of floats
        Non-exceedence probabilities that appear in the output confidence
        values dataframes and figures. Number of values should equal that of
        conf_heads otherwise an error is raised before infilling.
        Default is [0.05, 0.25, 0.50, 0.75, 0.95].
    fin_conf_head: unicode
        Label of the non-exceedence probabiliy that is used as the final infill
        value in the output dataframe. If should exist in conf_heads.
        Default is the third value of conf_heads i.e. \'var_0.5\'
    adj_probs_bounds: list of two floats
        The minimum and maximum cutoff cummulative probabilities. Before and
        after which all values of the range of the conditional probability
        are dropped.
        Default is [0.0001, 0.9999].
    flag_probs: list of two floats
        The non-exceedence probabilities for the conditional probability below
        and above which a value is flagged as suspicious.
        Default is [0.05, 0.95].
    n_round: integer
        The number of decimals to round all outputs to.
        Default is 3.
    cop_bins: integer
        The number of bins to use while plotting empirical copulas.
        Default is 20.
    max_corr: float
        The maximum correlation that any two stations can have while infilling,
        beyond which the station is not used.
        Default is 0.995
    ks_alpha: float
        The confidence limits of the Kolmogorov-Smirnov test. Should be between
        0 and 1.
        Default is 0.05.
    n_rand_infill_values: integer
        Create n_rand_infill_values output dataframes. Each of which contains
        a randomly selected value from the conditional distribution for a given
        step.
        Default is 0.
    debug_mode_flag: bool
        Turns multiprocessing off and allows for better debugging. The script
        has to run in the python debugger though.
        Default is False.
    plot_diag_flag: bool
        Plot and save outputs of each step for several variables. This allows
        for a detailed analysis of the output. This runs pretty slow.
        Default is False.
    plot_step_cdf_pdf_flag: bool
        Plot the conditional CDF and PDF for each step.
        Default is False.
    compare_infill_flag: bool
        Plot a comparison of infilled and observed values wherever they
        exist.
        Default is True.
    flag_susp_flag: bool
        Plot and save data flagging results.
        Default is False.
    force_infill_flag: bool
        Infill, if the number of available neighbors with valid values is
        atleast 1.
        Default is True.
    plot_neighbors_flag: bool
        Plot the neighbors for each infill station based.
        Default is False.
    take_min_stns_flag: bool
        Take n_nrst_stns_min if they are more than this.
        Default is True.
    overwrite_flag:
        Overwrite an output if it exists otherwise incorporate it.
        Default is True.
    read_pickles_flag: bool
        Read the neigboring stations correlations pickle. This avoids
        calculating the neighboring stations again. Valid only if
        nrst_stns_type is \'rank\' or \'symm\'.
        Default is False.
    use_best_stns_flag: bool
        Find a combination of stations that have the maximum number of stations
        available with values >= min_valid_vals while infilling a given
        station.
        Default is True.
    dont_stop_flag: bool
        Continue infilling even if an error occurs at a given step.
        Default is True.
    plot_long_term_corrs_flag: bool
        After selecting neighbors, plot the neighbors w.r.t descending rank
        correlation for a given infill station. Valid only if nrst_stns_type is
        \'rank\' of \'symm\'.
        Default is False.
    save_step_vars_flag: bool
        Save values of several parameters for each step during infilling.
        Default is False.
    plot_rand_flag: bool
        Plot the n_rand_infill_values with the non-exeedence values as well.
        Default is False.

    Outputs
    -------
    All the outputs mentioned here are created inside the \'out_dir\'.

    infill_var_df.csv: DataFrame, text
        The infilled dataframe. Looks the same as input but with infilled
        values inplace of missing values.
    infilled_flag_var_df.csv: DataFrame, text
        If flagging is True, this dataframe holds the flags for observed
        values, if they are within or out of bounds. 0 means within, -1 means
        below and +1 means above.
    n_avail_stns_df.csv: DataFrame, text
        The number of valid values available per step before and after
        infilling. Created by calling the  cmpt_plot_avail_stns method.
    n_avail_stns_compare.png: figure
        Plot of the n_avail_stns_df.csv. Saved by calling the
        cmpt_plot_avail_stns method.
    neighbor_stns_plots: directory
        If plot_neighbors_flag is True and the cmpt_plot_nrst_stns is called
        then plots for each infill stations' neighbors are saved in this
        directory.
    rank_corr_stns_plots: directory
        If plot_neighbors_flag is True and the cmpt_plot_rank_corr_stns is
        called then plots for each infill stations' neighbors are saved in this
        directory.
    long_term_correlations: directory
        If plot_long_term_corrs_flag is True and nrst_stns_type is \'rank\' or
        \'symm\' then the long-term correlation plots of each infill station
        with other stations are saved in this directory.
    summary_df.csv: DataFrame, text
        Save a dataframe showing some before and after infilling statistics.
    summary_df.png: Figure
        If the plot_summary method is called save the summary_df.csv as a
        figure.

    Methods
    -------
    infill:
        Infill the missing values using the criteria mentioned above.
    plot_ecops:
        Plot the empirical copulas of each infill station against its
        neighbors.
    cmpt_plot_avail_stns:
        Plot a figure and save a dataframe of the number of stations having
        valid values before and after infilling.
    cmpt_plot_stats:
        Plot statistics of all the stations in the input dataframe as a table.
    plot_summary:
        Plot the summary_df.csv as a colored table.

    Reference
    ---------
    Andras Bardossy, Geoffrey Pegram, Infilling missing precipitation records
    - A comparison of a new copula-based method with other techniques,
    Journal of Hydrology, Volume 519, Part A, 27 November 2014, Pages 1162-1170
    '''

    def __init__(self,
                 in_var_file,
                 in_coords_file,
                 out_dir,
                 infill_stns,
                 min_valid_vals,
                 infill_interval_type,
                 infill_type,
                 infill_dates_list,
                 n_nrst_stns_min=1,
                 n_nrst_stns_max=1,
                 ncpus=1,
                 skip_stns=None,
                 sep=';',
                 time_fmt='%Y-%m-%d',
                 freq='D',
                 verbose=True):


        # save all variables to a file with a timestamp
        self.verbose = bool(verbose)
        self.in_var_file = unicode(in_var_file)
        self.out_dir = unicode(out_dir)
        self.infill_stns = infill_stns
        self.infill_interval_type = unicode(infill_interval_type)
        self.infill_type = unicode(infill_type)
        self.infill_dates_list = list(infill_dates_list)
        self.min_valid_vals = int(min_valid_vals)
        self.in_coords_file = unicode(in_coords_file)
        self.n_nrst_stns_min = int(n_nrst_stns_min)
        self.n_nrst_stns_max = int(n_nrst_stns_max)
        self.ncpus = int(ncpus)
        self.skip_stns = list(skip_stns)
        self.sep = unicode(sep)
        self.time_fmt = unicode(time_fmt)
        self.freq = freq

        self.in_var_df = read_csv(self.in_var_file, sep=self.sep,
                                  index_col=0, encoding='utf-8')

        self.in_var_df.index = to_datetime(self.in_var_df.index,
                                           format=self.time_fmt)

        # Checking validity of parameters and adjustments if necessary

        assert self.in_var_df.shape[0] > 0, '\'in_var_df\' has no records!'
        assert self.in_var_df.shape[1] > 1, '\'in_var_df\' has < 2 fields!'
        self.in_var_df_orig = self.in_var_df.copy()

        self.in_var_df.columns = map(unicode, self.in_var_df.columns)

        if not os_exists(self.out_dir):
            os_mkdir(self.out_dir)

        if self.verbose:
            print 'INFO: \'in_var_df\' original shape:', self.in_var_df.shape

        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        if self.verbose:
            print 'INFO: \'in_var_df\' shape after dropping NaN steps:', \
                   self.in_var_df.shape

        self.in_coords_df = read_csv(self.in_coords_file, sep=sep, index_col=0,
                                     encoding='utf-8')
        assert self.in_coords_df.shape[0] > 0, \
            '\'in_coords_df\' has no records!'
        assert self.in_coords_df.shape[1] >= 2, \
            '\'in_coords_df\' has < 2 fields!'

        self.in_coords_df.index = map(unicode, self.in_coords_df.index)
        self.in_coords_df = \
            self.in_coords_df[~self.in_coords_df.index.duplicated(keep='last')]

        assert u'X' in self.in_coords_df.columns, \
            'Column \'X\' not in \'in_coords_df\'!'
        assert u'Y' in self.in_coords_df.columns, \
            'Column \'Y\' not in \'in_coords_df\'!'

        if self.verbose:
            print ('INFO: \'in_coords_df\' original shape:',
                   self.in_coords_df.shape)

        if hasattr(self.skip_stns, '__iter__'):
            if len(self.skip_stns) > 0:
                self.in_var_df.drop(labels=self.skip_stns,
                                    axis=1,
                                    inplace=True,
                                    errors='ignore')

                self.in_coords_df.drop(labels=self.skip_stns,
                                       axis=1,
                                       inplace=True,
                                       errors='ignore')

        assert self.min_valid_vals >= 1, \
            '\'min_valid_vals\' cannot be less than one!'

        assert self.n_nrst_stns_min >= 1, \
            '\'n_nrst_stns_min\' cannot be < one!'

        if self.n_nrst_stns_max + 1 > self.in_var_df.shape[1]:
            self.n_nrst_stns_max = self.in_var_df.shape[1] - 1
            print ('WARNING: \'n_nrst_stns_max\' reduced to %d' %
                   self.n_nrst_stns_max)

        assert self.n_nrst_stns_min <= self.n_nrst_stns_max, \
            '\'n_nrst_stns_min\' > \'n_nrst_stns_max\'!'

        assert (self.infill_type == u'discharge') or \
               (self.infill_type == u'precipitation'), \
            '\'infill_type\' can either be \'discharge\' or \'precipitation\'!'

        assert isinstance(self.ncpus, int), '\'ncpus\' not an integer!'
        assert self.ncpus >= 1, '\'ncpus\' cannot be less than one!'

        if (self.infill_interval_type == 'slice') or \
           (self.infill_interval_type == 'indiv'):
            assert hasattr(self.infill_dates_list, '__iter__'), \
               '\'infill_dates_list\' not an iterable!'

        if self.infill_interval_type == 'slice':
            assert len(self.infill_dates_list) == 2, \
                ('For infill_interval_type \'slice\' only '
                 'two objects inside \'infill_dates_list\' are allowed!')

            self.infill_dates_list = to_datetime(self.infill_dates_list,
                                                 format=self.time_fmt)
            assert self.infill_dates_list[1] > self.infill_dates_list[0], \
                'Infill dates not in ascending order!'
            self.infill_dates = date_range(start=self.infill_dates_list[0],
                                           end=self.infill_dates_list[-1],
                                           format=self.time_fmt,
                                           freq=self.freq)

            self.infill_dates = \
                self.in_var_df.index.intersection(self.infill_dates)

            assert self.infill_dates.shape[0] > 0, \
                ('The specified date slice has no days with any records '
                 'to work with!')
        elif self.infill_interval_type == 'all':
            self.infill_dates_list = None
            self.infill_dates = self.in_var_df.index
        elif self.infill_interval_type == 'indiv':
            assert len(self.infill_dates_list) > 0, \
                   '\'infill_dates_list\' is empty!'
            self.infill_dates = to_datetime(self.infill_dates_list,
                                            format=self.time_fmt)
        else:
            assert False, \
                ('\'infill_interval_type\' can only be \'slice\', \'all\', '
                 'or \'indiv\'!')

        insuff_val_cols = self.in_var_df.columns[self.in_var_df.count() <
                                                 self.min_valid_vals]

        if len(insuff_val_cols) > 0:
            self.in_var_df.drop(labels=insuff_val_cols, axis=1, inplace=True)
            if self.verbose:
                print (('INFO: The following stations (n=%d) '
                        'are with insufficient values:\n') %
                       insuff_val_cols.shape[0], insuff_val_cols.tolist())

        self.in_var_df.dropna(axis=(0, 1), how='all', inplace=True)

        if self.verbose:
            print ('INFO: \'in_var_df\' shape after dropping values less than '
                   '\'min_valid_vals\':', self.in_var_df.shape)

        assert self.min_valid_vals <= self.in_var_df.shape[0], \
            ('Number of stations in \'in_var_df\' less than '
             '\'min_valid_vals\' after dropping days with insufficient '
             'records!')

        commn_stns = intersect1d(self.in_var_df.columns,
                                 self.in_coords_df.index)
        self.in_var_df = self.in_var_df[commn_stns]

        self.in_coords_df = self.in_coords_df.loc[commn_stns]
        self.xs = self.in_coords_df['X'].values
        self.ys = self.in_coords_df['Y'].values

        if self.infill_stns == 'all':
            self.infill_stns = self.in_var_df.columns

        if self.verbose:
            print ('INFO: \'in_var_df\' shape after station name intersection '
                   'with \'in_coords_df\':', self.in_var_df.shape)
            print ('INFO: \'in_coords_df\' shape after station name '
                   'intersection with \'in_var_df\':', self.in_coords_df.shape)

        assert self.n_nrst_stns_min < self.in_var_df.shape[1], \
            ('Number of stations in \'in_var_df\' less than '
             '\'n_nrst_stns_min\' after intersecting station names!')
        if self.n_nrst_stns_max >= self.in_var_df.shape[1]:
            self.n_nrst_stns_max = self.in_var_df.shape[1] - 1
            print (('WARNING: \'n_nrst_stns_max\' set to %d after station '
                    'names intersection!') % self.n_nrst_stns_max)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
                (('Station %s not in input variable dataframe '
                  'anymore!') % infill_stn)

        self.n_infill_stns = len(self.infill_stns)
        self.infill_stns = Index(self.infill_stns)

        self.infill_dates = \
            self.infill_dates.intersection(self.in_var_df.index)

        # check if atleast one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
            ('No infill dates exist in \'in_var_df\' after dropping '
             'stations and records with insufficient information!')

        ### Initiating additional required variables
        self.nrst_stns_list = []
        self.nrst_stns_dict = {}

        self.rank_corr_stns_list = []
        self.rank_corr_stns_dict = {}

        if self.infill_type == 'precipitation':
            self.nrst_stns_type = 'rank'
        else:
            self.nrst_stns_type = 'symm'  # can be rank or dist or symm

        self.n_discret = 300
        self.n_norm_symm_flds = 100

        self.fig_size_long = (20, 7)
        self.out_fig_dpi = 150
        self.out_fig_fmt = 'png'

        self.conf_heads = ['var_0.05',
                           'var_0.25',
                           'var_0.50',
                           'var_0.75',
                           'var_0.95']
        self.conf_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        self.fin_conf_head = self.conf_heads[2]
        self.adj_prob_bounds = [0.0001, 0.9999]
        self.flag_probs = [0.05, 0.95]
        self.n_round = 3
        self.cop_bins = 20
        self.max_corr = 0.995
        self.ks_alpha = 0.05
        self.n_rand_infill_values = 0

        self._norm_cop_pool = None

        self._infilled = False
        self._dist_cmptd = False
        self._rank_corr_cmptd = False
        self._conf_ser_cmptd = False
        self._bef_all_chked = False
        self._bef_infill_chked = False

        self.debug_mode_flag = False
        self.plot_diag_flag = False
        self.plot_step_cdf_pdf_flag = False
        self.compare_infill_flag = True
        self.flag_susp_flag = False
        self.force_infill_flag = True
        self.plot_neighbors_flag = False
        self.take_min_stns_flag = True
        self.overwrite_flag = True
        self.read_pickles_flag = False
        self.use_best_stns_flag = True
        self.dont_stop_flag = False
        self.plot_long_term_corrs_flag = False
        self.save_step_vars_flag = False
        self.plot_rand_flag = False

        self.out_var_file = os_join(self.out_dir, 'infilled_var_df.csv')
        self.out_flag_file = os_join(self.out_dir, 'infilled_flag_var_df.csv')
        self.out_stns_avail_file = os_join(self.out_dir, 'n_avail_stns_df.csv')
        self.out_stns_avail_fig = os_join(self.out_dir, 'n_avail_stns_compare')
        self.out_nebor_plots_dir = os_join(self.out_dir, 'neighbor_stns_plots')
        self.out_rank_corr_plots_dir = os_join(self.out_dir,
                                               'rank_corr_stns_plots')
        self.out_long_term_corrs_dir = os_join(self.out_dir,
                                               'long_term_correlations')

        self.out_summary_file = os_join(self.out_dir, 'summary_df.csv')
        self.out_summary_fig = os_join(self.out_dir, 'summary_df')
        self._out_rank_corrs_pkl_file = os_join(self.out_dir,
                                                'rank_corrs_mat.pkl')
        self._out_rank_corrs_ctr_pkl_file = os_join(self.out_dir,
                                                    'rank_corrs_ctr_mat.pkl')

        if self.infill_type == u'precipitation':
            self.var_le_trs = 0.0
            self.var_ge_trs = 1.0
            self.ge_le_trs_n = 1

        self.summary_df = None  # defined in _before_infill_checks method
        return

    def cmpt_plot_nrst_stns(self):
        '''
        Plot nearest stations around each infill station
        '''
        if self.verbose:
            print 'INFO: Computing and plotting nearest stations...'

        ### cmpt nrst stns
        for infill_stn in self.infill_stns:
            # get the x and y coordinates of the infill_stn
            infill_x, infill_y = \
                self.in_coords_df[['X', 'Y']].loc[infill_stn].values

            # calculate distances of all stations from the infill_stn
            dists = vectorize(get_dist)(infill_x, infill_y, self.xs, self.ys)
            dists_df = DataFrame(index=self.in_var_df.columns, data=dists,
                                 columns=['dists'], dtype=float)
            dists_df.sort_values('dists', axis=0, inplace=True)

            # take the nearest n_nrn stations to the infill_stn
            for nrst_stn in dists_df.iloc[:self.n_nrst_stns_max + 1].index:
                if nrst_stn not in self.nrst_stns_list:
                    self.nrst_stns_list.append(nrst_stn)

            # put the neighboring stations in a dictionary for each infill_stn
            self.nrst_stns_dict[infill_stn] = \
                dists_df.iloc[1:self.n_nrst_stns_max + 1].index

            _ = self.nrst_stns_dict[infill_stn]
            assert len(_) >= self.n_nrst_stns_min, \
                (('Neighboring stations less than \'n_nrst_stns_min\' '
                  'for station: %s') % infill_stn)

        # have the nrst_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.nrst_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
                (('station %s not in input variable dataframe '
                  'anymore!') % infill_stn)

        # check if at least one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
            ('None of the infill dates exist in \'in_var_df\' after dropping '
             'stations and records with insufficient information!')

        if self.verbose:
            print ('INFO: \'in_var_df\' shape after calling '
                   '\'cmpt_plot_nrst_stns\':', self.in_var_df.shape)

        ### plot nrst stns
        if self.plot_neighbors_flag:
            if not os_exists(self.out_nebor_plots_dir):
                os_mkdir(self.out_nebor_plots_dir)

            tick_font_size = 5
            for infill_stn in self.infill_stns:
                (infill_x,
                 infill_y) = self.in_coords_df[['X',
                                                'Y']].loc[infill_stn].values
                _nebs = self.nrst_stns_dict[infill_stn]
                _lab = ('neibor_stn (%d)' %
                        self.nrst_stns_dict[infill_stn].shape[0])
                nrst_stns_ax = plt.subplot(111)
                nrst_stns_ax.scatter(infill_x,
                                     infill_y,
                                     c='r',
                                     label='infill_stn')
                nrst_stns_ax.scatter(self.in_coords_df['X'].loc[_nebs],
                                     self.in_coords_df['Y'].loc[_nebs],
                                     alpha=0.75,
                                     c='c',
                                     label=_lab)
                plt_texts = []
                _txt_obj = nrst_stns_ax.text(infill_x,
                                             infill_y,
                                             infill_stn,
                                             va='top',
                                             ha='left',
                                             fontsize=tick_font_size)
                plt_texts.append(_txt_obj)
                for stn in self.nrst_stns_dict[infill_stn]:
                    _txt_obj = nrst_stns_ax.text(
                        self.in_coords_df['X'].loc[stn],
                        self.in_coords_df['Y'].loc[stn],
                        stn,
                        va='top',
                        ha='left',
                        fontsize=5)
                    plt_texts.append(_txt_obj)

                adjust_text(plt_texts, only_move={'points': 'y',
                                                  'text': 'y'})
                nrst_stns_ax.grid()
                nrst_stns_ax.set_xlabel('Eastings', size=tick_font_size)
                nrst_stns_ax.set_ylabel('Northings', size=tick_font_size)
                nrst_stns_ax.legend(framealpha=0.5, loc=0)
                plt.setp(nrst_stns_ax.get_xticklabels(), size=tick_font_size)
                plt.setp(nrst_stns_ax.get_yticklabels(), size=tick_font_size)
                plt.savefig(os_join(self.out_nebor_plots_dir,
                                    '%s_neibor_stns.png' % infill_stn),
                            dpi=self.out_fig_dpi)
                plt.clf()
            plt.close()
        self._dist_cmptd = True
        return

    def cmpt_plot_rank_corr_stns(self):
        '''
        Plot stations around infill stations based on highest pearson
        rank correlations
        '''
        if self.verbose:
            print 'INFO: Computing highest correlation stations...'

        recalc_rank_corrs = False

        ### load rank_corr pickle
        if os_exists(self._out_rank_corrs_pkl_file) and self.read_pickles_flag:
            if self.verbose:
                print 'INFO: Loading rank correlations pickle...'

            self.rank_corrs_df = read_pickle(self._out_rank_corrs_pkl_file)
            self.rank_corrs_df = \
                self.rank_corrs_df.apply(lambda x: to_numeric(x))

            rank_corr_stns = self.rank_corrs_df.index

            if self.infill_stns.shape[0] != rank_corr_stns.shape[0]:
                recalc_rank_corrs = True
                if self.verbose:
                    print 'INFO: Rank correlations pickle is not up-to-date!'

            if not recalc_rank_corrs:
                for infill_stn in self.infill_stns:
                    if infill_stn not in rank_corr_stns:
                        recalc_rank_corrs = True
                        if self.verbose:
                            print ('INFO: Rank correlations pickle is not '
                                   'up-to-date!')
                        break

            if not recalc_rank_corrs:
                self.rank_corr_vals_ctr_df = \
                    read_pickle(self._out_rank_corrs_ctr_pkl_file)
                self.rank_corr_vals_ctr_df = \
                    self.rank_corr_vals_ctr_df.apply(lambda x: to_numeric(x))

        else:
            recalc_rank_corrs = True

        ### cmpt rank_corrs
        if recalc_rank_corrs:
            self.rank_corrs_df = DataFrame(index=self.infill_stns,
                                           columns=self.in_var_df.columns,
                                           dtype=float)

            self.rank_corrs_df = \
                self.rank_corrs_df.apply(lambda x: to_numeric(x))
            self.rank_corr_vals_ctr_df = self.rank_corrs_df.copy()

            drop_infill_stns = []
            drop_str = ('Station %s has no records available in the '
                        'given time period and is dropped!')

            self.n_infill_stns = self.infill_stns.shape[0]

            tot_corrs_written = 0
            for i_stn in self.infill_stns:
                ser_i = self.in_var_df.loc[:, i_stn].dropna().copy()
                ser_i_index = ser_i.index

                for j_stn in self.in_var_df.columns:
                    if i_stn == j_stn:
                        continue

                    try:
                        if not isnan(self.rank_corrs_df.loc[j_stn, i_stn]):
                            self.rank_corrs_df.loc[i_stn, j_stn] = \
                                self.rank_corrs_df.loc[j_stn, i_stn]
                            self.rank_corr_vals_ctr_df.loc[i_stn, j_stn] = \
                                self.rank_corr_vals_ctr_df.loc[j_stn, i_stn]
                            tot_corrs_written += 1
                            continue
                    except KeyError:
                        pass

                    ser_j = self.in_var_df.loc[:, j_stn].dropna().copy()
                    index_ij = ser_i_index.intersection(ser_j.index)

                    if index_ij.shape[0] <= self.min_valid_vals:
                        continue

                    new_ser_i = ser_i.loc[index_ij].copy()
                    new_ser_j = ser_j.loc[index_ij].copy()
                    prob_ser_i = \
                        (new_ser_i.rank() / (new_ser_i.shape[0] + 1.)).values
                    prob_ser_j = \
                        (new_ser_j.rank() / (new_ser_j.shape[0] + 1.)).values
                    correl = get_corrcoeff(prob_ser_i, prob_ser_j)

                    if self.nrst_stns_type == 'symm':
                        asymms_arr = full((self.n_norm_symm_flds, 2), nan)
                        for asymm_idx in xrange(self.n_norm_symm_flds):
                            as_1, as_2 = \
                                self._get_norm_rand_symms(correl)
                            asymms_arr[asymm_idx, 0] = as_1
                            asymms_arr[asymm_idx, 1] = as_2

                        assert np_all(isfinite(asymms_arr)), \
                            'Invalid values of asymmetries!'

                        min_as_1, max_as_1 = (asymms_arr[:, 0].min(),
                                              asymms_arr[:, 0].max())
                        min_as_2, max_as_2 = (asymms_arr[:, 1].min(),
                                              asymms_arr[:, 1].max())

                        act_asymms = get_asymms_sample(prob_ser_i, prob_ser_j)
                        act_as_1, act_as_2 = (act_asymms['asymm_1'],
                                              act_asymms['asymm_2'])

                        as_1_norm = False
                        as_2_norm = False
                        if (act_as_1 >= min_as_1) and (act_as_1 <= max_as_1):
                            as_1_norm = True
                        if (act_as_2 >= min_as_2) and (act_as_2 <= max_as_2):
                            as_2_norm = True

                        if not (as_1_norm and as_2_norm):
                            correl = nan

                    self.rank_corrs_df.loc[i_stn, j_stn] = correl
                    self.rank_corr_vals_ctr_df.loc[i_stn, j_stn] = \
                        new_ser_i.shape[0]

                    if not isnan(correl):
                        tot_corrs_written += 1

                if not self.rank_corrs_df.loc[i_stn].dropna().shape[0]:
                    if self.dont_stop_flag:
                        drop_infill_stns.append(i_stn)
                        if self.verbose:
                            print 'WARNING: ', drop_str % i_stn
                        continue
                    else:
                        raise RuntimeError(drop_str % i_stn)

            if self.verbose:
                print 'INFO: %d out of possible %d correlations written' % \
                      (tot_corrs_written,
                       (self.n_infill_stns * (self.in_var_df.shape[1] - 1)))

        self.rank_corrs_df.to_pickle(self._out_rank_corrs_pkl_file)
        self.rank_corr_vals_ctr_df.to_pickle(self._out_rank_corrs_ctr_pkl_file)

        self.rank_corrs_df.dropna(axis=(0, 1), how='all', inplace=True)
        self.rank_corr_vals_ctr_df.dropna(axis=(0, 1), how='all', inplace=True)

        for infill_stn in self.infill_stns:
            if not (infill_stn in self.rank_corrs_df.index):
                if self.dont_stop_flag:
                    print (('\a Warning: the station %s is not in '
                            '\'rank_corrs_df\' and is removed from the '
                            'infill stations list!') % infill_stn)
                    if infill_stn not in drop_infill_stns:
                        drop_infill_stns.append(infill_stn)
                else:
                    raise KeyError(('The station %s is not in '
                                    '\'rank_corrs_df\' anymore!') % infill_stn)

        for drop_stn in drop_infill_stns:
            self.infill_stns = self.infill_stns.drop(drop_stn)

        if not self.infill_stns.shape[0]:
            raise RuntimeError('No stations to work with!')

        bad_stns_list = []
        bad_stns_neighbors_count = []

        ### cmpt best rank_corr stns
        for infill_stn in self.infill_stns:
            if infill_stn not in self.rank_corr_stns_list:
                self.rank_corr_stns_list.append(infill_stn)
            stn_correl_ser = self.rank_corrs_df.loc[infill_stn].dropna().copy()

            if stn_correl_ser.shape[0] == 0:
                raise RuntimeError('No neighbors for station %s!' % infill_stn)
            stn_correl_ser[:] = where(isfinite(stn_correl_ser[:]),
                                      np_abs(stn_correl_ser[:]),
                                      nan)
            stn_correl_ser.sort_values(axis=0, ascending=False, inplace=True)

            # take the nearest n_nrn stations to the infill_stn
            for rank_corr_stn in stn_correl_ser.index:
                if rank_corr_stn not in self.rank_corr_stns_list:
                    self.rank_corr_stns_list.append(rank_corr_stn)

            # put the neighboring stations in a dictionary for each infill_stn
            self.rank_corr_stns_dict[infill_stn] = \
                stn_correl_ser.iloc[:self.n_nrst_stns_max].index

            curr_n_neighbors = len(self.rank_corr_stns_dict[infill_stn])
            if (curr_n_neighbors < self.n_nrst_stns_min) and \
               (not self.force_infill_flag):
                bad_stns_list.append(infill_stn)
                bad_stns_neighbors_count.append(curr_n_neighbors)

            if not self.force_infill_flag:
                assert curr_n_neighbors >= self.n_nrst_stns_min, \
                    (('Rank correlation stations (n=%d) less than '
                      '\'n_nrst_stns_min\' or no neighbor for station: '
                      '%s') % (curr_n_neighbors, infill_stn))

        if len(bad_stns_list) > 0:
            for bad_stn in bad_stns_list:
                self.infill_stns = self.infill_stns.drop(bad_stn)
                del self.rank_corr_stns_dict[bad_stn]

            self.rank_corrs_df.drop(labels=bad_stns_list, axis=0, inplace=True)
            self.rank_corr_vals_ctr_df.drop(labels=bad_stns_list,
                                            axis=0,
                                            inplace=True)

            if self.verbose:
                print 'INFO: These infill station(s) had too few values',
                print 'to be considered for the rest of the analysis:'
                print 'Station: n_neighbors'
                for bad_stn in zip(bad_stns_list, bad_stns_neighbors_count):
                    print '%s: %d' % (bad_stn[0], bad_stn[1])

        # have the rank_corr_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.rank_corr_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        if not self.in_var_df.shape[0]:
            raise RuntimeError('No dates to work with!')

        if not self.dont_stop_flag:
            for infill_stn in self.infill_stns:
                assert infill_stn in self.in_var_df.columns, \
                    ('infill station %s not in input variable dataframe '
                     'anymore!' % infill_stn)

            # check if at least one infill date is in the in_var_df
            date_in_dates = False
            full_dates = self.in_var_df.index
            for infill_date in self.infill_dates:
                if infill_date in full_dates:
                    date_in_dates = True
                    break

            assert date_in_dates, \
                ('None of the infill dates exist in \'in_var_df\' '
                 'after dropping stations and records with insufficient '
                 'information!')

        if self.verbose:
            print ('INFO: \'in_var_df\' shape after calling '
                   '\'cmpt_plot_rank_corr_stns\':', self.in_var_df.shape)

        ### plot rank_corr stns
        if self.plot_neighbors_flag:
            if not os_exists(self.out_rank_corr_plots_dir):
                os_mkdir(self.out_rank_corr_plots_dir)

            tick_font_size = 5
            for infill_stn in self.infill_stns:
                infill_x, infill_y = \
                    self.in_coords_df[['X', 'Y']].loc[infill_stn].values

                _nebs = self.rank_corr_stns_dict[infill_stn]
                _n_nebs = self.rank_corr_stns_dict[infill_stn].shape[0]
                hi_corr_stns_ax = plt.subplot(111)
                hi_corr_stns_ax.scatter(infill_x,
                                        infill_y,
                                        c='r',
                                        label='infill_stn')
                hi_corr_stns_ax.scatter(self.in_coords_df['X'].loc[_nebs],
                                        self.in_coords_df['Y'].loc[_nebs],
                                        alpha=0.75,
                                        c='c',
                                        label='hi_corr_stn (%d)' % _n_nebs)
                plt_texts = []
                _txt_obj = hi_corr_stns_ax.text(infill_x,
                                                infill_y,
                                                infill_stn,
                                                va='top',
                                                ha='left',
                                                fontsize=tick_font_size)
                plt_texts.append(_txt_obj)
                for stn in self.rank_corr_stns_dict[infill_stn]:
                    _lab = '%s\n(%0.2f)' % (stn,
                                            self.rank_corrs_df.loc[infill_stn,
                                                                   stn])
                    if not infill_stn == stn:
                        _txt_obj = hi_corr_stns_ax.text(
                            self.in_coords_df['X'].loc[stn],
                            self.in_coords_df['Y'].loc[stn],
                            _lab,
                            va='top',
                            ha='left',
                            fontsize=5)
                        plt_texts.append(_txt_obj)

                adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
                hi_corr_stns_ax.grid()
                hi_corr_stns_ax.set_xlabel('Eastings', size=tick_font_size)
                hi_corr_stns_ax.set_ylabel('Northings', size=tick_font_size)
                hi_corr_stns_ax.legend(framealpha=0.5, loc=0)
                plt.setp(hi_corr_stns_ax.get_xticklabels(),
                         size=tick_font_size)
                plt.setp(hi_corr_stns_ax.get_yticklabels(),
                         size=tick_font_size)
                plt.savefig(os_join(self.out_rank_corr_plots_dir,
                                    'rank_corr_stn_%s.png' % infill_stn),
                            dpi=self.out_fig_dpi)
                plt.clf()

        plt.close()

        ### plot long-term rank_corrs
        if self.plot_long_term_corrs_flag:
            if self.verbose:
                print 'INFO: Plotting long-term correlation neighbors'

            if not os_exists(self.out_long_term_corrs_dir):
                os_mkdir(self.out_long_term_corrs_dir)

            for infill_stn in self.infill_stns:
                tick_font_size = 6
                curr_nebs = self.rank_corr_stns_dict[infill_stn]

                corrs_arr = self.rank_corrs_df.loc[infill_stn,
                                                   curr_nebs].values
                corrs_ctr_arr = \
                    self.rank_corr_vals_ctr_df.loc[infill_stn,
                                                   curr_nebs].values
                corrs_ctr_arr[isnan(corrs_ctr_arr)] = 0

                n_stns = corrs_arr.shape[0]
                fig, corrs_ax = plt.subplots(1, 1, figsize=(1.0 * n_stns, 3))
                corrs_ax.matshow(corrs_arr.reshape(1, n_stns),
                                 vmin=0,
                                 vmax=2,
                                 cmap=cmaps.Blues,
                                 origin='lower')
                for s in range(n_stns):
                    corrs_ax.text(s,
                                  0,
                                  '%0.4f\n(%d)' % (corrs_arr[s],
                                                   int(corrs_ctr_arr[s])),
                                  va='center',
                                  ha='center',
                                  fontsize=tick_font_size)

                corrs_ax.set_yticks([])
                corrs_ax.set_yticklabels([])
                corrs_ax.set_xticks(range(0, n_stns))
                corrs_ax.set_xticklabels(curr_nebs)

                corrs_ax.spines['left'].set_position(('outward', 10))
                corrs_ax.spines['right'].set_position(('outward', 10))
                corrs_ax.spines['top'].set_position(('outward', 10))
                corrs_ax.spines['bottom'].set_position(('outward', 10))

                corrs_ax.tick_params(labelleft=False,
                                     labelbottom=True,
                                     labeltop=True,
                                     labelright=False)

                plt.setp(corrs_ax.get_xticklabels(),
                         size=tick_font_size,
                         rotation=45)
                plt.suptitle('station: %s long-term corrs' % infill_stn)
                _ = 'long_term_stn_%s_rank_corrs.png' % infill_stn
                plt.savefig(os_join(self.out_long_term_corrs_dir, _),
                            dpi=self.out_fig_dpi)
                plt.close()

        self._rank_corr_cmptd = True
        return

    def cmpt_plot_stats(self):
        '''
        Compute and plot statistics of each station
        '''
        if self.verbose:
            print 'INFO: Computing and plotting input variable statistics...'

        stats_cols = ['min', 'max', 'mean', 'stdev',
                      'CoV', 'skew', 'count']
        self.stats_df = DataFrame(index=self.in_var_df.columns,
                                  columns=stats_cols,
                                  dtype=float)

        for i, stn in enumerate(self.stats_df.index):
            curr_ser = self.in_var_df[stn].dropna().copy()
            curr_min = curr_ser.min()
            curr_max = curr_ser.max()
            curr_mean = curr_ser.mean()
            curr_stdev = curr_ser.std()
            curr_coovar = curr_stdev / curr_mean
            curr_skew = curr_ser.skew()
            curr_count = curr_ser.count()
            self.stats_df.iloc[i] = [curr_min,
                                     curr_max,
                                     curr_mean,
                                     curr_stdev,
                                     curr_coovar,
                                     curr_skew,
                                     curr_count]

        self.stats_df = self.stats_df.apply(lambda x: to_numeric(x))

        tick_font_size = 5
        stats_arr = self.stats_df.values
        n_stns = stats_arr.shape[0]
        n_cols = stats_arr.shape[1]

        fig, stats_ax = plt.subplots(1,
                                     1,
                                     figsize=(0.45 * n_stns, 0.8 * n_cols))
        stats_ax.matshow(stats_arr.T,
                         cmap=cmaps.Blues,
                         vmin=0,
                         vmax=0,
                         origin='upper')

        for s in zip(repeat(range(n_stns), n_cols),
                     tile(range(n_cols), n_stns)):
            stats_ax.text(s[0],
                          s[1],
                          ('%0.2f' % stats_arr[s[0], s[1]]).rstrip('0'),
                          va='center',
                          ha='center',
                          fontsize=tick_font_size)

        stats_ax.set_xticks(range(0, n_stns))
        stats_ax.set_xticklabels(self.stats_df.index)
        stats_ax.set_yticks(range(0, n_cols))
        stats_ax.set_yticklabels(self.stats_df.columns)

        stats_ax.spines['left'].set_position(('outward', 10))
        stats_ax.spines['right'].set_position(('outward', 10))
        stats_ax.spines['top'].set_position(('outward', 10))
        stats_ax.spines['bottom'].set_position(('outward', 10))

        stats_ax.set_xlabel('Stations', size=tick_font_size)
        stats_ax.set_ylabel('Statistics', size=tick_font_size)

        stats_ax.tick_params(labelleft=True,
                             labelbottom=True,
                             labeltop=True,
                             labelright=True)

        plt.setp(stats_ax.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(stats_ax.get_yticklabels(), size=tick_font_size)

        plt.savefig(os_join(self.out_dir, 'var_statistics.png'),
                    dpi=self.out_fig_dpi)
        plt.close()
        return

    def plot_ecops(self):
        '''
        Plot empirical copulas of each station against all other
        '''
        self.ecops_dir = os_join(self.out_dir, 'empirical_copula_plots')
        if not os_exists(self.ecops_dir):
            os_mkdir(self.ecops_dir)

        self._get_ncpus()

        if not self._dist_cmptd:
            self.cmpt_plot_nrst_stns()
            assert self._dist_cmptd, 'Call \'cmpt_plot_nrst_stns\' first!'

        if self.nrst_stns_type == 'dist':
            pass
        elif (self.nrst_stns_type == 'rank') or \
             (self.nrst_stns_type == 'symm'):
            if not self._rank_corr_cmptd:
                self.cmpt_plot_rank_corr_stns()
                assert self._rank_corr_cmptd, \
                    'Call \'cmpt_plot_rank_corr_stns\' first!'
        else:
            assert False, 'Incorrect \'nrst_stns_type\': %s' % \
                   str(self.nrst_stns_type)

        if self.verbose:
            print ('INFO: Plotting empirical copulas of infill stations '
                   'against others...')

        if (self.ncpus == 1) or self.debug_mode_flag:
            self._plot_ecops(self.infill_stns)
        else:
            idxs = linspace(0, len(self.infill_stns), self.ncpus + 1,
                            endpoint=True, dtype='int64')
            sub_cols = []
            if self.infill_stns.shape[0] <= self.ncpus:
                sub_cols = [[stn] for stn in self.infill_stns]
            else:
                for idx in range(self.ncpus):
                    sub_cols.append(self.infill_stns[idxs[idx]:idxs[idx + 1]])

            try:
                list(self._norm_cop_pool.uimap(self._plot_ecops, sub_cols))
                self._norm_cop_pool.clear()
            except:
                self._norm_cop_pool.close()
                self._norm_cop_pool.join()
                raise RuntimeError(('Failed to execute \'plot_ecops\' '
                                    'successfully!'))
        return

    def infill(self):
        '''
        Perform the infilling based on given data
        '''
        if not self._bef_infill_chked:
            self._before_infill_checks()
        assert self._bef_infill_chked, 'Call \'_before_infill_checks\' first!'

        if self.debug_mode_flag and self.dont_stop_flag:
            self.dont_stop_flag = False
            if self.verbose:
                print 'INFO: \'dont_stop_flag\' set to False!'

        if self.verbose:
            print '\n'
            print 'INFO: Flags:'
            print '  \a debug_mode_flag:', self.debug_mode_flag
            print '  \a plot_diag_flag:', self.plot_diag_flag
            print '  \a plot_step_cdf_pdf_flag:', self.plot_step_cdf_pdf_flag
            print '  \a compare_infill_flag:', self.compare_infill_flag
            print '  \a flag_susp_flag:', self.flag_susp_flag
            print '  \a force_infill_flag:', self.force_infill_flag
            print '  \a plot_neighbors_flag:', self.plot_neighbors_flag
            print '  \a take_min_stns_flag:', self.take_min_stns_flag
            print '  \a overwrite_flag:', self.overwrite_flag
            print '  \a read_pickles_flag:', self.read_pickles_flag
            print '  \a use_best_stns_flag:', self.use_best_stns_flag
            print '  \a dont_stop_flag:', self.dont_stop_flag
            print '  \a plot_long_term_corrs_flag:',
            print self.plot_long_term_corrs_flag
            print '  \a save_step_vars_flag:', self.save_step_vars_flag
            print '  \a plot_rand_flag:', self.plot_rand_flag
            print '\n'

            print 'INFO: Infilling...'
            print 'INFO: Infilling type is:', self.infill_type
            print (('INFO: Maximum records to process per station: '
                    '%d') % self.infill_dates.shape[0])
            print (('INFO: Total number of stations to infill: '
                    '%d') % len(self.infill_stns))
            if self.n_rand_infill_values:
                print 'INFO: n_rand_infill_values:', self.n_rand_infill_values

            print '\n'

        lw = 0.8
        alpha = 0.7

        for ii, infill_stn in enumerate(self.infill_stns):
            if self.verbose:
                print (('  \a Going through station %d of %d: '
                        '%s') % (ii + 1, self.n_infill_stns, infill_stn))

            self.curr_infill_stn = infill_stn
            self.stn_out_dir = os_join(self.out_dir, infill_stn)
            out_conf_df_file = \
                os_join(self.stn_out_dir,
                        'stn_%s_infill_conf_vals_df.csv' % infill_stn)
            out_add_info_file = \
                os_join(self.stn_out_dir,
                        'add_info_df_stn_%s.csv' % infill_stn)

            ### load infill
            no_out = True
            if (not self.overwrite_flag) and os_exists(out_conf_df_file):
                if self.verbose:
                    print '    \a Output exists already. Not overwriting.'

                try:
                    out_conf_df = read_csv(out_conf_df_file,
                                           sep=str(self.sep),
                                           encoding='utf-8',
                                           index_col=0)
                    out_conf_df.index = to_datetime(out_conf_df.index,
                                                    format=self.time_fmt)

                    if not self.n_rand_infill_values:
                        _idxs = \
                            isnan(self.in_var_df_orig.loc[self.infill_dates,
                                                          self.curr_infill_stn
                                                          ])
                        _ser = self.in_var_df_orig.loc[self.infill_dates,
                                                       self.curr_infill_stn]
                        out_stn_ser = _ser.where(
                            logical_not(_idxs),
                            out_conf_df[self.fin_conf_head],
                            axis=0)
                        self.out_var_df.loc[out_conf_df.index,
                                            infill_stn] = out_stn_ser
                    else:
                        for rand_idx in range(self.n_rand_infill_values):
                            _idxs = isnan(
                                self.in_var_df_orig.loc[
                                        self.infll_dates,
                                        self.curr_infill_stn])
                            _idxs = logical_not(_idxs)
                            _ser = self.in_var_df_orig.loc[
                                self.infill_dates, self.curr_infill_stn]
                            _lab = self.fin_conf_head % rand_idx
                            out_stn_ser = _ser.where(_idxs,
                                                     out_conf_df[_lab],
                                                     axis=0)
                            self.out_var_dfs_list[rand_idx].loc[
                                out_conf_df.index, infill_stn] = out_stn_ser

                    no_out = False
                except Exception as msg:
                    raise RuntimeError(('Error while trying to read and '
                                        'update values from the existing '
                                        'dataframe:' + msg))

            if not (self.overwrite_flag or no_out):
                continue

            if not self.compare_infill_flag:
                nan_idxs = where(
                    isnan(self.in_var_df.loc[
                        self.infill_dates, self.curr_infill_stn].values))[0]
            else:
                nan_idxs = range(self.infill_dates.shape[0])

            n_nan_idxs = len(nan_idxs)
            if self.verbose:
                print '    \a %d steps to infill' % n_nan_idxs

            if (n_nan_idxs == 0) and (not self.compare_infill_flag):
                continue

            if self.nrst_stns_type == 'rank':
                self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
            elif self.nrst_stns_type == 'dist':
                self.curr_nrst_stns = self.nrst_stns_dict[infill_stn]
            elif self.nrst_stns_type == 'symm':
                self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
            else:
                assert False, 'Incorrect \'nrst_stns_type\'!'

            ### mkdirs
            dir_list = [self.stn_out_dir]
            if self.plot_step_cdf_pdf_flag:
                self.stn_infill_cdfs_dir = os_join(self.stn_out_dir,
                                                   'stn_infill_cdfs')
                self.stn_infill_pdfs_dir = os_join(self.stn_out_dir,
                                                   'stn_infill_pdfs')
                dir_list.extend([self.stn_infill_cdfs_dir,
                                 self.stn_infill_pdfs_dir])

            if self.plot_diag_flag:
                self.stn_step_cdfs_dir = os_join(self.stn_out_dir,
                                                 'stn_step_cdfs')
                self.stn_step_corrs_dir = os_join(self.stn_out_dir,
                                                  'stn_step_corrs')
                dir_list.extend([self.stn_step_cdfs_dir,
                                 self.stn_step_corrs_dir])

            if self.save_step_vars_flag:
                self.stn_step_vars_dir = os_join(self.stn_out_dir,
                                                 'stn_step_vars')
                dir_list.extend([self.stn_step_vars_dir])

            for _dir in dir_list:
                if not os_exists(_dir):
                    os_mkdir(_dir)

            idxs = linspace(0,
                            n_nan_idxs,
                            self.ncpus + 1,
                            endpoint=True,
                            dtype='int64')

            if self.verbose:
                infill_start = timeit.default_timer()

            ### initiate infill
            if ((idxs.shape[0] == 1) or
                    (self.ncpus == 1) or
                    self.debug_mode_flag):
                out_conf_df, out_add_info_df = self._infill(self.infill_dates)
                _ser = self.in_var_df_orig.loc[self.infill_dates,
                                               self.curr_infill_stn]
                _idxs = isnan(self.in_var_df_orig.loc[self.infill_dates,
                                                      self.curr_infill_stn])
                _idxs = logical_not(_idxs)
                if not self.n_rand_infill_values:
                    out_stn_ser = (_ser.where(_idxs,
                                              out_conf_df[self.fin_conf_head],
                                              axis=0)).copy()
                    self.out_var_df.loc[out_conf_df.index,
                                        infill_stn] = out_stn_ser
                else:
                    for rand_idx in range(self.n_rand_infill_values):
                        _lab = self.fin_conf_head % rand_idx
                        out_stn_ser = (_ser.where(_idxs,
                                                  out_conf_df[_lab],
                                                  axis=0)).copy()
                        self.out_var_dfs_list[rand_idx].loc[
                                out_conf_df.index, infill_stn] = out_stn_ser
            else:
                n_sub_dates = 0
                sub_infill_dates_list = []
                for idx in range(self.ncpus):
                    sub_dates = self.infill_dates[
                                nan_idxs[idxs[idx]:idxs[idx + 1]]]
                    sub_infill_dates_list.append(sub_dates)
                    n_sub_dates += sub_dates.shape[0]

                assert n_sub_dates == n_nan_idxs, \
                    (('\'n_sub_dates\' (%d) and \'self.infill_dates\' '
                      '(%d) of unequal length!') %
                        (n_sub_dates, self.infill_dates.shape[0]))

                try:
                    sub_dfs = list(self._norm_cop_pool.uimap(
                                   self._infill, sub_infill_dates_list))
                    self._norm_cop_pool.clear()
                except:
                    self._norm_cop_pool.close()
                    self._norm_cop_pool.join()
                    return RuntimeError

                out_conf_df = DataFrame(index=self.infill_dates,
                                        columns=self.conf_ser.index,
                                        dtype=float)
                out_add_info_df = DataFrame(index=self.infill_dates,
                                            dtype=float,
                                            columns=['infill_status',
                                                     'n_neighbors_raw',
                                                     'n_neighbors_fin',
                                                     'act_val_prob'])
                for sub_df in sub_dfs:
                    sub_conf_df = sub_df[0]
                    sub_add_info_df = sub_df[1]

                    _ser = self.in_var_df_orig.loc[sub_conf_df.index,
                                                   self.curr_infill_stn]
                    _idxs = isnan(self.in_var_df_orig.loc[
                                  sub_conf_df.index, self.curr_infill_stn])
                    _idxs = logical_not(_idxs)

                    if not self.n_rand_infill_values:
                        sub_stn_ser = (_ser.where(_idxs,
                                       sub_conf_df[self.fin_conf_head],
                                       axis=0)).copy()
                        self.out_var_df.loc[sub_conf_df.index,
                                            infill_stn] = sub_stn_ser
                    else:
                        for rand_idx in range(self.n_rand_infill_values):
                            _lab = self.fin_conf_head % rand_idx
                            sub_stn_ser = (_ser.where(_idxs,
                                                      sub_conf_df[_lab],
                                                      axis=0)).copy()
                            self.out_var_dfs_list[rand_idx].loc[
                                    sub_conf_df.index,
                                    infill_stn] = sub_stn_ser

                    out_conf_df.update(sub_conf_df)
                    out_add_info_df.update(sub_add_info_df)

            n_infilled_vals = out_conf_df.dropna().shape[0]
            if self.verbose:
                print '    \a %d steps infilled' % n_infilled_vals

                infill_stop = timeit.default_timer()
                fin_secs = infill_stop - infill_start
                print (('    \a Took %0.3f secs, %0.3e secs per '
                        'step') % (fin_secs, fin_secs / n_nan_idxs))

            ### prepare output
            out_conf_df = out_conf_df.apply(lambda x: to_numeric(x))
            out_conf_df.to_csv(out_conf_df_file,
                               sep=str(self.sep),
                               encoding='utf-8')
            out_add_info_df.to_csv(out_add_info_file,
                                   sep=str(self.sep),
                                   encoding='utf-8')

            self.summary_df.loc[self.curr_infill_stn, self._av_vals_lab] = \
                self.in_var_df[self.curr_infill_stn].dropna().shape[0]

            self.summary_df.loc[self.curr_infill_stn,
                                self._miss_vals_lab] = n_nan_idxs

            self.summary_df.loc[self.curr_infill_stn,
                                self._infilled_vals_lab] = n_infilled_vals

            self.summary_df.loc[
                    self.curr_infill_stn, self._max_avail_nebs_lab] = \
                len(self.curr_nrst_stns)

            self.summary_df.loc[self.curr_infill_stn,
                                self._avg_avail_nebs_lab] = \
                round(out_add_info_df['n_neighbors_fin'].dropna().mean(), 1)

            ### make plots
            # plot number of gauges available and used
            plt.figure(figsize=self.fig_size_long)
            infill_ax = plt.subplot(111)
            infill_ax.plot(self.infill_dates,
                           out_add_info_df['n_neighbors_raw'].values,
                           label='n_neighbors_raw',
                           c='r',
                           alpha=alpha,
                           lw=lw+0.5,
                           ls='-')
            infill_ax.plot(self.infill_dates,
                           out_add_info_df['n_neighbors_fin'].values,
                           label='n_neighbors_fin',
                           c='b',
                           marker='o',
                           lw=0,
                           ms=2)

            infill_ax.set_xlabel('Time')
            infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
            infill_ax.set_ylabel('Stations used')
            infill_ax.set_ylim(-1, infill_ax.get_ylim()[1] + 1)

            plt.suptitle(('Number of raw available and finally used stations '
                          'for infilling at station: %s') % infill_stn)
            plt.grid()
            plt.legend(framealpha=0.5, loc=0)
            plt.savefig(os_join(self.stn_out_dir,
                                'stns_used_infill_%s.png' % infill_stn),
                        dpi=self.out_fig_dpi)
            plt.close()

            # the original unfilled series of the infilled station
            act_var = self.in_var_df_orig[
                      infill_stn].loc[self.infill_dates].values

            # plot the infilled series
            out_infill_plot_loc = os_join(self.stn_out_dir,
                                          'missing_infill_%s.png' % infill_stn)
            plot_infill_cond = True

            if not self.overwrite_flag:
                plot_infill_cond = plot_infill_cond and \
                                   (not os_exists(out_infill_plot_loc))

            use_mp = not (self.debug_mode_flag or (self.ncpus == 1))
            compare_iter = None
            flag_susp_iter = None

            infill_start = timeit.default_timer()

            if plot_infill_cond:
                print '    \a Plotting infill...'
                args_tup = (act_var, out_conf_df, out_infill_plot_loc)
                if use_mp:
                    self._norm_cop_pool.uimap(self._plot_infill_ser,
                                              (args_tup, ))
                else:
                    self._plot_infill_ser(args_tup)

            # plot the comparison between the actual and infilled series
            out_compar_plot_loc = os_join(self.stn_out_dir,
                                          'compare_infill_%s.png' % infill_stn)
            plot_compar_cond = self.compare_infill_flag
            if not self.overwrite_flag:
                plot_compar_cond = plot_compar_cond and \
                                   (not os_exists(out_compar_plot_loc))

            if plot_compar_cond and np_any(act_var):
                print '    \a Plotting infill comparison...'
                args_tup = (act_var,
                            out_conf_df,
                            out_compar_plot_loc,
                            out_add_info_df)

                if use_mp:
                    compare_iter = self._norm_cop_pool.uimap(
                                   self._plot_compar_ser, (args_tup, ))
                else:
                    self.summary_df.update(self._plot_compar_ser(args_tup))
            else:
                print '    \a Nothing to compare...'

            # plot steps showing if the actual data is within the bounds
            out_flag_susp_loc = os_join(self.stn_out_dir,
                                        'flag_infill_%s.png' % infill_stn)
            plot_flag_susp_cond = self.flag_susp_flag
            if not self.overwrite_flag:
                plot_flag_susp_cond = plot_flag_susp_cond and \
                                      (not os_exists(out_flag_susp_loc))

            if plot_flag_susp_cond and np_any(act_var):
                print '    \a Plotting infill flags...'
                args_tup = (act_var, out_conf_df, out_flag_susp_loc)
                if use_mp:
                    flag_susp_iter = \
                        self._norm_cop_pool.uimap(self._plot_flag_susp_ser,
                                                  (args_tup, ))
                else:
                    _ = self._plot_flag_susp_ser(args_tup)
                    self.summary_df.update(_[0])
                    self.flag_df.update(_[1])

            else:
                print '    \a Nothing to flag...'

            if use_mp:
                self._norm_cop_pool.clear()

            infill_stop = timeit.default_timer()
            fin_secs = infill_stop - infill_start
            print '    \a Took %0.3f secs for plotting other stuff' % fin_secs

            if compare_iter:
                self.summary_df.update(list(compare_iter)[0])
            if flag_susp_iter:
                _ = list(flag_susp_iter)[0]
                self.summary_df.update(_[0])
                self.flag_df.update(_[1])

        if not self.n_rand_infill_values:
            self.out_var_df.to_csv(self.out_var_file,
                                   sep=str(self.sep),
                                   encoding='utf-8')
        else:
            out_var_file_tmpl = self.out_var_file[:-4]
            for rand_idx in range(self.n_rand_infill_values):
                out_var_file = out_var_file_tmpl + '_%0.4d.csv' % rand_idx
                self.out_var_dfs_list[rand_idx].to_csv(out_var_file,
                                                       sep=str(self.sep),
                                                       encoding='utf-8')

        self.summary_df.loc[:, self._compr_lab].fillna(0.0, inplace=True)
        self.summary_df.to_csv(self.out_summary_file,
                               sep=str(self.sep),
                               encoding='utf-8')

        if self.flag_susp_flag:
            self.flag_df.to_csv(self.out_flag_file,
                                sep=str(self.sep),
                                encoding='utf-8',
                                float_format='%2.0f')
        self._infilled = True
        print '\n'
        return

    def cmpt_plot_avail_stns(self):
        '''
        To compare the number of stations before and after infilling
        '''
        if self.verbose:
            print ('INFO: Computing and plotting number of stations '
                   'available per step...')

        assert self._infilled, 'Call \'do_infill\' first!'

        self.avail_nrst_stns_orig_ser = self.in_var_df_orig.count(axis=1)

        if not self.n_rand_infill_values:
            self.avail_nrst_stns_ser = self.out_var_df.count(axis=1)
        else:
            self.avail_nrst_stns_ser = self.out_var_dfs_list[0].count(axis=1)

        assert self.avail_nrst_stns_orig_ser.sum() > 0, 'in_var_df is empty!'
        assert self.avail_nrst_stns_ser.sum() > 0, 'out_var_df is empty!'

        plt.figure(figsize=self.fig_size_long)
        plt.plot(self.avail_nrst_stns_orig_ser.index,
                 self.avail_nrst_stns_orig_ser.values,
                 alpha=0.8, label='Original')
        plt.plot(self.avail_nrst_stns_ser.index,
                 self.avail_nrst_stns_ser.values,
                 alpha=0.8, label='Infilled')
        plt.xlabel('Time')
        plt.ylabel('Number of stations with valid values')
        plt.legend(framealpha=0.5)
        plt.grid()
        plt.savefig(self.out_stns_avail_fig, dpi=self.out_fig_dpi)
        plt.close()

        out_index = self.avail_nrst_stns_ser.index.union(
                    self.avail_nrst_stns_orig_ser.index)
        fin_df = DataFrame(index=out_index,
                           columns=['original', 'infill'],
                           dtype=float)
        fin_df['original'] = self.avail_nrst_stns_orig_ser
        fin_df['infill'] = self.avail_nrst_stns_ser
        fin_df.to_csv(self.out_stns_avail_file,
                      sep=str(self.sep), encoding='utf-8')
        return

    def plot_summary(self):
        '''
        Plot the summary_df as a table with formatting
        '''

        if self.verbose:
            print 'INFO: Plotting summary table...'

        assert self._infilled, 'Call \'do_infill\' first!'

        clr_alpha = 0.4
        n_segs = 100
        font_size = 6

        max_columns_per_fig = 15

        cmap_name = 'my_clr_list'
        ks_lim_label = (('%%age values within %0.0f%% '
                         'KS-limits') % (100 * (1.0 - self.ks_alpha)))

        r_2_g_colors = [(1, 0, 0, clr_alpha), (0, 1, 0, clr_alpha)]  # R -> G
        r_2_g_cm = LinearSegmentedColormap.from_list(cmap_name,
                                                     r_2_g_colors,
                                                     N=n_segs)
        g_2_r_cm = LinearSegmentedColormap.from_list(
                   cmap_name, list(reversed(r_2_g_colors)), N=n_segs)

        clr_log_norm = LogNorm(0.001, 1)

        n_stns = self.summary_df.shape[0]

        if n_stns > max_columns_per_fig:
            n_figs = ceil(n_stns / float(max_columns_per_fig))
            cols_per_fig_idxs = unique(linspace(0,
                                                n_stns,
                                                n_figs + 1,
                                                dtype=int))
        else:
            cols_per_fig_idxs = array([0, n_stns])

        for stn_idx in range(cols_per_fig_idxs.shape[0] - 1):
            curr_summary_df = self.summary_df.iloc[
                    cols_per_fig_idxs[stn_idx]:cols_per_fig_idxs[stn_idx+1],
                    :].copy()

            colors_df = DataFrame(index=curr_summary_df.index,
                                  columns=curr_summary_df.columns,
                                  dtype=object)

            ### available values
            avail_vals = curr_summary_df.loc[:, self._av_vals_lab]
            n_max_available = avail_vals.max(skipna=True)
            n_min_available = avail_vals.min(skipna=True)

            if isnan(n_max_available):
                print 'WARNING: No summary to plot!'
                return

            if n_max_available == n_min_available:
                n_min_available = 0.0
            avail_vals[isnan(avail_vals)] = n_min_available
            _avail_vals_rats = (avail_vals.values - n_min_available) / \
                               (n_max_available + 0.0 - n_min_available)
            available_val_clrs = r_2_g_cm(_avail_vals_rats)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._av_vals_lab] = available_val_clrs[i]

            ### missing values
            miss_vals = curr_summary_df.loc[:, self._miss_vals_lab]
            n_max_missing = miss_vals.max(skipna=True)
            n_min_missing = miss_vals.min(skipna=True)
            if n_max_missing == n_min_missing:
                n_min_missing = 0.0
            miss_vals[isnan(miss_vals)] = n_min_missing
            _miss_val_rats = (miss_vals.values - n_min_missing) / \
                             (n_max_missing + 0.0 - n_min_missing)
            missing_val_clrs = g_2_r_cm(_miss_val_rats)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._miss_vals_lab] = missing_val_clrs[i]

            ### infilled values
            infill_vals = curr_summary_df.loc[:,
                                              self._infilled_vals_lab].values
            infill_vals[isnan(infill_vals)] = 0.0
            infilled_val_clrs = r_2_g_cm(infill_vals / miss_vals.values)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._infilled_vals_lab] = \
                    infilled_val_clrs[i]

            ### max available neighbors
            max_avail_neb_vals = \
                curr_summary_df.loc[:, self._max_avail_nebs_lab].values
            if self.n_nrst_stns_max == self.n_nrst_stns_min:
                min_nebs = 0.0
            else:
                min_nebs = self.n_nrst_stns_min
            max_avail_neb_vals[isnan(max_avail_neb_vals)] = min_nebs
            _manvrs = (max_avail_neb_vals - min_nebs) / \
                      (self.n_nrst_stns_max + 0.0 - min_nebs)
            max_nebs_clrs = r_2_g_cm(_manvrs)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._max_avail_nebs_lab] = max_nebs_clrs[i]

            ### average used neighbors
            _aanrs = \
                curr_summary_df.loc[:, self._avg_avail_nebs_lab].values / \
                max_avail_neb_vals
            avg_nebs_clrs = r_2_g_cm(_aanrs)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._avg_avail_nebs_lab] = avg_nebs_clrs[i]

            ### compared
            compared_vals = \
                curr_summary_df.loc[:, self._compr_lab].values / \
                curr_summary_df.loc[:, 'Infilled values'].values
            compared_vals[isnan(compared_vals)] = 0.0
            n_compare_clrs = r_2_g_cm(compared_vals)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._compr_lab] = n_compare_clrs[i]

            ### KS limits
            for i, stn in enumerate(curr_summary_df.index):
                if (curr_summary_df.loc[stn, ks_lim_label] >=
                        (100 * (1.0 - self.ks_alpha))):
                    ks_clr = r_2_g_colors[1]
                else:
                    ks_clr = r_2_g_colors[0]

                colors_df.loc[stn, ks_lim_label] = ks_clr

            ### flagged
            for i, stn in enumerate(colors_df.index):
                if (curr_summary_df.loc[stn, self._flagged_lab] >
                   (100 * (1 - self.flag_probs[1] + self.flag_probs[0]))):
                    flag_clr = r_2_g_colors[0]
                else:
                    flag_clr = r_2_g_colors[1]

                colors_df.loc[stn, self._flagged_lab] = flag_clr

            ### bias
            max_bias = curr_summary_df.loc[:, self._bias_lab].max(skipna=True)
            min_bias = curr_summary_df.loc[:, self._bias_lab].min(skipna=True)
            max_bias = max(abs(max_bias), abs(min_bias))
            if max_bias == 0.0:
                max_bias = 1.0
            if isnan(max_bias):
                max_bias = 1
            bias_vals = fabs(curr_summary_df.loc[:, self._bias_lab].values)
            bias_vals[isnan(bias_vals)] = max_bias
            bias_vals = clr_log_norm(bias_vals / (max_bias + 0.0))
            bias_clrs = g_2_r_cm(bias_vals.data)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._bias_lab] = bias_clrs[i]

            ### mean absolute error
            max_mae = curr_summary_df.loc[:, self._mae_lab].max(skipna=True)
            if max_mae == 0.0:
                max_mae = 1.0
            if isnan(max_mae):
                max_mae = 1.0
            mae_vals = fabs(curr_summary_df.loc[:, self._mae_lab].values)
            mae_vals[isnan(mae_vals)] = max_mae
            mae_vals = clr_log_norm(mae_vals / (max_mae + 0.0))
            mae_clrs = g_2_r_cm(mae_vals.data)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._mae_lab] = mae_clrs[i]

            ### rmse
            max_rmse = curr_summary_df.loc[:, self._rmse_lab].max(skipna=True)
            if max_rmse == 0.0:
                max_rmse = 1.0
            if isnan(max_rmse):
                max_rmse = 1.0
            rmse_vals = fabs(curr_summary_df.loc[:, self._rmse_lab].values)
            rmse_vals[isnan(rmse_vals)] = max_rmse
            rmse_vals = clr_log_norm(rmse_vals / (max_rmse + 0.0))
            rmse_clrs = g_2_r_cm(rmse_vals.data)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._rmse_lab] = rmse_clrs[i]

            ### nse
            nse_vals = curr_summary_df.loc[:, self._nse_lab].copy().values
            nse_vals[isnan(nse_vals)] = 0.0
            nse_clrs = r_2_g_cm(where(nse_vals < 0.0, 0, nse_vals))
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._nse_lab] = nse_clrs[i]

            ### ln_nse
            ln_nse_vals = curr_summary_df.loc[:,
                                              self._ln_nse_lab].copy().values
            ln_nse_vals[isnan(ln_nse_vals)] = 0.0
            ln_nse_clrs = r_2_g_cm(where(ln_nse_vals < 0.0, 0.0, ln_nse_vals))
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._ln_nse_lab] = ln_nse_clrs[i]

            ### kge
            kge_vals = curr_summary_df.loc[:, self._kge_lab].copy().values
            kge_vals[isnan(kge_vals)] = 0.0
            kge_clrs = r_2_g_cm(where(kge_vals < 0.0, 0.0, kge_vals))
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._kge_lab] = kge_clrs[i]

            ### pcorr
            pcorr_vals = fabs(curr_summary_df.loc[:, self._pcorr_lab].values)
            pcorr_vals[isnan(pcorr_vals)] = 0.0
            pcorr_clrs = r_2_g_cm(pcorr_vals)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._pcorr_lab] = pcorr_clrs[i]

            ### scorr
            scorr_vals = fabs(curr_summary_df.loc[:, self._scorr_lab].values)
            scorr_vals[isnan(scorr_vals)] = 0.0
            scorr_clrs = r_2_g_cm(scorr_vals)
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._scorr_lab] = scorr_clrs[i]

            ### means and variances
            for i, stn in enumerate(colors_df.index):
                colors_df.loc[stn, self._mean_obs_lab] = r_2_g_colors[-1]
                colors_df.loc[stn, self._mean_infill_lab] = r_2_g_colors[-1]
                colors_df.loc[stn, self._var_obs_lab] = r_2_g_colors[-1]
                colors_df.loc[stn, self._var_infill_lab] = r_2_g_colors[-1]

            ### plot the table
            _fs = (2 + (0.5 * curr_summary_df.shape[0]), 6)
            plt_fig = plt.figure(figsize=_fs)

            col_labs = curr_summary_df.index
            row_labs = curr_summary_df.columns

            table_text_list = curr_summary_df.values.T

            row_colors = [['0.75'] * 4] * row_labs.shape[0]
            col_colors = [['0.75'] * 4] * col_labs.shape[0]

            table_ax = plt.table(cellText=table_text_list,
                                 bbox=[0, 0, 1, 1],
                                 rowLabels=row_labs,
                                 colLabels=col_labs,
                                 rowLoc='right',
                                 colLoc='left',
                                 cellLoc='right',
                                 rowColours=row_colors,
                                 colColours=col_colors,
                                 cellColours=array(colors_df[:].T))

            table_ax.auto_set_font_size(False)
            table_ax.set_fontsize(font_size)

            # adjust header text if too wide
            renderer = plt_fig.canvas.get_renderer()
            table_cells = table_ax.get_celld()

            max_text_width = 0
            cell_tups = table_cells.keys()
            for cell_tup in cell_tups:
                if cell_tup[0] == 0:
                    curr_text_width = table_cells[cell_tup].get_text()
                    curr_text_width = \
                        curr_text_width.get_window_extent(renderer)
                    curr_text_width = curr_text_width.width
                    if curr_text_width > max_text_width:
                        max_text_width = curr_text_width

            table_cells = table_ax.get_celld()
            padding = table_cells[(0, 0)].PAD
            cell_width = \
                float(table_cells[(0, 0)].get_window_extent(renderer).width)
            cell_width = cell_width - (2. * padding * cell_width)

            new_font_size = font_size * cell_width / max_text_width

            if new_font_size < font_size:
                cell_tups = table_cells.keys()
                for cell_tup in cell_tups:
                    if cell_tup[0] == 0:
                        table_cells[cell_tup].set_fontsize(new_font_size)

            plt.title('Normal Copula Infilling Summary', loc='right')
            plt.axis('off')
            out_fig = self.out_summary_fig + '_%0.2d' % (stn_idx + 1)
            plt.savefig(out_fig, dpi=self.out_fig_dpi, bbox_inches='tight')
        return

    def _get_ncpus(self):
        '''
        Set the number of processes to be used

        Call it in the first line of any function that has mp in it
        '''
        if self.debug_mode_flag:
            self.ncpus = 1
            if self.dont_stop_flag:
                self.dont_stop_flag = False
                if self.verbose:
                    print 'INFO: \'dont_stop_flag\' set to False!'

        elif not hasattr(self._norm_cop_pool, '_id'):
            self._norm_cop_pool = mp_pool(nodes=self.ncpus)
        return

    def _cmpt_conf_ser(self):
        '''
        Check if all the variables for the calculation of confidence intervals
        are correct
        '''
        try:
            self.conf_probs = array(self.conf_probs, dtype=float)
        except:
            raise ValueError(('Some or all \'_conf_probs\' are '
                              'non-float: %s' % str(self.conf_probs)))

        try:
            self.adj_prob_bounds = array(self.adj_prob_bounds, dtype=float)
        except:
            raise ValueError(('Some or all \'_adj_prob_bounds\' are '
                              'non-float: %s' % str(self.adj_prob_bounds)))

        try:
            self.conf_heads = array(self.conf_heads, dtype=unicode)
        except:
            raise ValueError(('Some or all \'_conf_heads\' are '
                              'non-unicode: %s' % str(self.conf_heads)))

        try:
            self.flag_probs = array(self.flag_probs, dtype=float)
        except:
            raise ValueError(('Some or all \'_flag_probs\' are '
                              'non-float: %s' % str(self.flag_probs)))

        assert self.conf_probs.shape[0] >= 2, \
            '\'_conf_probs\' cannot be less than 2 values!'
        assert self.conf_probs.shape == self.conf_heads.shape, \
            (('\'_conf_probs\' (%s) and \'_conf_heads\' (%s) cannot have '
              'different shapes!' % (self.conf_probs.shape,
                                     self.conf_heads.shape)))
        assert isfinite(np_all(self.conf_probs)), \
            'Invalid values in \'_conf_probs\': %s' % str(self.conf_probs)

        assert np_any(where(ediff1d(self.conf_probs) > 0, 1, 0)), \
            '\'_conf_probs\' not ascending (%s)!' % str(self.conf_probs)

        assert max(self.conf_probs) <= self.adj_prob_bounds[1], \
            ('max \'adj_prob_bounds\' < max \'_conf_probs\' (%s)!' %
             str(self.adj_prob_bounds))

        assert min(self.conf_probs) > self.adj_prob_bounds[0], \
            ('min \'adj_prob_bounds\' > min \'_conf_probs\' (%s)!' %
             str(self.adj_prob_bounds))

        assert isfinite(np_all(self.flag_probs)), \
            'Invalid values in \'_flag_probs\': %s' % str(self.flag_probs)

        assert isfinite(np_all(self.adj_prob_bounds)), \
            (('Invalid values in '
              '\'_adj_prob_bounds\': %s') % str(self.adj_prob_bounds))

        assert len(self.adj_prob_bounds) == 2, \
            ('\'adj_bounds_probs\' are not two values (%s)!' %
             str(self.adj_prob_bounds))

        assert self.adj_prob_bounds[0] < self.adj_prob_bounds[1], \
            (('\'adj_bounds_probs\' not ascending (%s)!') %
             str(self.adj_prob_bounds))

        if self.n_rand_infill_values:
            # changing this would have effects
            # this would appear as rand_%0.4d in the output
            self.fin_conf_head = 'rand_%0.4d'
        else:
            assert self.fin_conf_head in self.conf_heads, \
                '\'_fin_conf_head\': %s not in \'_conf_heads\': %s' % \
                (self.fin_conf_head, self.conf_heads)

        if self.verbose:
            print ('INFO: Using \'%s\' as the final infill value' %
                   str(self.fin_conf_head))

        if self.flag_susp_flag:
            assert len(self.flag_probs) == 2, \
                'Only two values allowed inside \'_flag_probs\'!'
            assert self.flag_probs[0] < self.flag_probs[1], \
                ('\'_flags_probs\': first value should be smaller '
                 'than the last!')

            for _flag_val in self.flag_probs:
                assert np_any(isclose(_flag_val, self.conf_probs)), \
                    ('\'_flag_prob\' (%s) value not in \'_conf_probs\' (%s)!' %
                     (str(_flag_val), str(self.conf_probs)))

        self.conf_ser = Series(index=self.conf_heads,
                               data=self.conf_probs,
                               dtype=float)

        if self.n_rand_infill_values:
            for i in range(self.n_rand_infill_values):
                self.conf_ser['rand_%0.4d' % i] = nan

        self._conf_ser_cmptd = True
        return

    def _get_norm_rand_symms(self, corr):
        rv_1 = norm_ppf_py_arr(gen_n_rns_arr(self.n_norm_symm_flds))
        rv_2 = norm_ppf_py_arr(gen_n_rns_arr(self.n_norm_symm_flds))
        rv_3 = (corr * rv_1) + ((1 - corr**2)**0.5 * rv_2)

        rv_1 = rankdata(rv_1) / (self.n_norm_symm_flds + 1.0)
        rv_3 = rankdata(rv_3) / (self.n_norm_symm_flds + 1.0)
        asymms = get_asymms_sample(rv_1, rv_3)
        return asymms['asymm_1'], asymms['asymm_2']

    def _before_all_checks(self):
        assert isinstance(self.n_discret, int), '\'n_discret\' is non-integer!'
        assert isinstance(self.n_round, int), '\'n_round\' is non-integer!'
        assert isinstance(self.cop_bins, int), '\'cop_bins\' is non-integer!'
        assert isinstance(self.max_corr, float), '\'max_corr\' is non-float!'
        assert isinstance(self.ks_alpha, float), '\'ks_alpha\' is non-float!'
        assert isinstance(self.out_fig_dpi, int), \
            '\'out_fig_dpi\' is non-integer!'
        assert isinstance(self.out_fig_fmt, unicode), \
            '\'out_fig_dpi\' is non-unicode!'
        assert isinstance(self.n_norm_symm_flds, int), \
            '\'n_norm_symm_flds\' is a non-integer!'
        assert isinstance(self.n_rand_infill_values, int), \
            '\'n_rand_infill_values\' is non-integer'

        assert self.n_discret > 0, '\'n_discret\' less than 1!'
        assert self.n_norm_symm_flds > 0, '\'n_norm_symm_flds\' less than 1!'
        assert self.out_fig_dpi > 0, '\'out_fig_dpi\' less than 1!'
        assert self.n_round >= 0, '\'n_round\' less than 0!'
        assert self.cop_bins > 0, '\'cop_bins\' less than 1!'
        assert (self.max_corr >= 0) and (self.max_corr <= 1.0), \
            '\'max_corr\' (%s) not between 0 and 1' % str(self.max_corr)
        assert (self.ks_alpha > 0) and (self.ks_alpha < 1.0), \
            '\'ks_alpha\' (%s) not between 0 and 1' % str(self.ks_alpha)

        if self.n_rand_infill_values:
            _c1 = self.n_rand_infill_values > 0
            _c2 = self.n_rand_infill_values < 10000
            assert _c1 and _c2, \
                (('\'n_rand_infill_values\' (%s) not between '
                  '0 and 10000') % str(self.n_rand_infill_values))

        if self.infill_type == 'precipitation':
            assert self.nrst_stns_type != 'symm', \
                '\'infill_type\' cannot be \'symm\' for precipitation!'

        assert hasattr(self.fig_size_long, '__iter__'), \
            '\'fig_size_long\' not an iterable!'
        assert len(self.fig_size_long) == 2, \
            'Only two values allowed inside \'fig_size_long\'!'

        fig = plt.figure()
        supp_fmts = fig.canvas.get_supported_filetypes().keys()
        assert self.out_fig_fmt in supp_fmts, \
            ('\'out_fig_fmt\' (%s) not in the supported formats list (%s)' %
             (self.out_fig_fmt, supp_fmts))

        plt.close()
        self._bef_all_chked = True
        return

    def _before_infill_checks(self):
        self._before_all_checks()
        assert self._bef_all_chked, 'Call \'_before_all_checks\' first!'

        if self.plot_diag_flag:
            self.compare_infill_flag = True
            self.force_infill_flag = True
            self.plot_step_cdf_pdf_flag = True
            self.flag_susp_flag = True
            self.save_step_vars_flag = True

        if not self._dist_cmptd:
            self.cmpt_plot_nrst_stns()
            assert self._dist_cmptd, 'Call \'cmpt_plot_nrst_stns\' first!'

        if (self.nrst_stns_type == 'rank') or (self.nrst_stns_type == 'symm'):
            if self.nrst_stns_type == 'rank':
                if self.verbose:
                    print 'INFO: using RANKS to get neighboring stations'

            elif self.nrst_stns_type == 'symm':
                if self.verbose:
                    print ('INFO: using RANKS with SYMMETRIES to get '
                           'neighboring stations')

            if not self._rank_corr_cmptd:
                self.cmpt_plot_rank_corr_stns()
                assert self._rank_corr_cmptd, \
                    'Call \'cmpt_plot_rank_corr_stns\' first!'

        elif self.nrst_stns_type == 'dist':
            if self.verbose:
                print 'INFO: using DISTANCE to get neighboring stations'

        else:
            assert False, 'Incorrect \'nrst_stns_type\': (%s)' % \
                   str(self.nrst_stns_type)

        self.n_infill_stns = self.infill_stns.shape[0]
        assert self.n_infill_stns, 'No stations to work with!'

        _ = self.in_var_df[self.infill_stns]
        self.infill_dates = _.index.intersection(self.infill_dates)
        self.n_infill_dates = self.infill_dates.shape[0]
        assert self.n_infill_dates, 'No dates to work with!'

        if self.verbose:
            print 'INFO: Final infill_stns count = %d' % self.n_infill_stns
            print 'INFO: Final infill_dates count = %d' % self.n_infill_dates

        self._cmpt_conf_ser()
        assert self._conf_ser_cmptd, 'Call \'_cmpt_conf_ser\' first!'

        if self.infill_type == u'precipitation':
            assert isinstance(self.var_le_trs, float), \
                '\'var_le_trs\' is non-float!'
            assert isinstance(self.var_ge_trs, float), \
                '\'var_ge_trs\' is non-float!'
            assert isinstance(self.ge_le_trs_n, int), \
                '\'ge_le_trs_n\' is non-integer!'

            assert self.var_le_trs <= self.var_ge_trs, \
                '\'var_le_trs\' > \'var_ge_trs\'!'
            assert self.ge_le_trs_n > 0, \
                '\'self.ge_le_trs_n\' less than 1!'
        else:
            self.var_le_trs, self.var_ge_trs, self.ge_le_trs_n = 3*[None]

        if self.flag_susp_flag:
            self.flag_df = DataFrame(columns=self.infill_stns,
                                     index=self.infill_dates,
                                     dtype=float)
            self.compare_infill_flag = True

        self._get_ncpus()

        if not self.n_rand_infill_values:
            self.out_var_df = self.in_var_df_orig.copy()
            self.out_var_dfs_list = None
        else:
            self.out_var_dfs_list = [self.in_var_df_orig.copy()] * \
                                    self.n_rand_infill_values
            self.out_var_df = None

        self._av_vals_lab = 'Available values'
        self._miss_vals_lab = 'Missing values'
        self._infilled_vals_lab = 'Infilled values'
        self._max_avail_nebs_lab = 'Max. available neighbors'
        self._avg_avail_nebs_lab = 'Avg. neighbors used for infilling'
        self._compr_lab = 'Compared values'
        self._ks_lims_lab = (('%%age values within %0.0f%% '
                              'KS-limits') % (100 * (1.0 - self.ks_alpha)))
        self._flagged_lab = 'Flagged values %age'
        self._mean_obs_lab = 'Mean (observed)'
        self._mean_infill_lab = 'Mean (infilled)'
        self._var_obs_lab = 'Variance (observed)'
        self._var_infill_lab = 'Variance (infilled)'
        self._bias_lab = 'Bias'
        self._mae_lab = 'Mean abs. error'
        self._rmse_lab = 'Root mean sq. error'
        self._nse_lab = 'NSE'
        self._ln_nse_lab = 'Ln-NSE'
        self._kge_lab = 'KGE'
        self._pcorr_lab = 'Pearson correlation'
        self._scorr_lab = 'Spearman correlation'

        self._summary_cols = [self._av_vals_lab,
                              self._miss_vals_lab,
                              self._infilled_vals_lab,
                              self._max_avail_nebs_lab,
                              self._avg_avail_nebs_lab,
                              self._compr_lab,
                              self._ks_lims_lab,
                              self._flagged_lab,
                              self._mean_obs_lab,
                              self._mean_infill_lab,
                              self._var_obs_lab,
                              self._var_infill_lab,
                              self._bias_lab,
                              self._mae_lab,
                              self._rmse_lab,
                              self._nse_lab,
                              self._ln_nse_lab,
                              self._kge_lab,
                              self._pcorr_lab,
                              self._scorr_lab]

        self.summary_df = DataFrame(index=self.infill_stns,
                                    columns=self._summary_cols,
                                    dtype=float)

        self._bef_infill_chked = True
        return

    def _full_tb(self, sys_info):
        exc_type, exc_value, exc_traceback = sys_info
        tb_fmt_obj = format_exception(exc_type, exc_value, exc_traceback)
        for trc in tb_fmt_obj:
            print trc
        if not self.dont_stop_flag:
            raise Exception
        return

    def _infill(self, infill_dates):
        try:
            out_conf_df, out_add_info_df = self.__infill(infill_dates)
            return out_conf_df, out_add_info_df
        except:
            plt.close()
            self._full_tb(exc_info())
            if self.dont_stop_flag:
                return out_conf_df, out_add_info_df
        return

    def __infill(self, infill_dates):
        def _get_best_stns(best_stns):
            '''
            Select stations based on maximum number of common available steps
            while they are greater than min_valid_vals

            Time to infill increases with increase in n_nrn_min and
            n_nrn_max if use_best_stns_flag is True
            '''
            max_count = 0
            curr_count = 0

            for i in range(curr_var_df.shape[1] - 1, 0, -1):
                if (not self.force_infill_flag) and (i < self.n_nrst_stns_min):
                    break

                combs = combinations(avail_cols_raw, i)

                for comb in combs:
                    _ = [self.curr_infill_stn] + list(comb)
                    curr_df = curr_var_df[_].dropna(axis=0)
                    curr_count = curr_df.shape[0]
                    if (curr_count > max_count) and \
                       (curr_count > self.min_valid_vals) and \
                       (self.curr_infill_stn in curr_df.columns):
                        max_count = curr_count
                        best_stns = curr_df.columns[1:]
                if curr_count > self.min_valid_vals:
                    break

            return best_stns

        def _create_cdfs_ftns_and_plots():
            # create probability and standard normal value dfs
            probs_df = DataFrame(index=curr_var_df.index,
                                 columns=curr_var_df.columns,
                                 dtype=float)
            norms_df = probs_df.copy()

            py_del = nan
            py_zero = nan

            for col in curr_var_df.columns:
                # CDFs
                var_ser = curr_var_df[col].copy()

                if self.infill_type == u'precipitation':
                    # get probability of zero and below threshold
                    zero_idxs = where(var_ser.values == self.var_le_trs)[0]
                    zero_prob = float(zero_idxs.shape[0]) / var_ser.shape[0]

                    thresh_idxs = where(
                                  logical_and(var_ser.values > self.var_le_trs,
                                              var_ser.values <=
                                              self.var_ge_trs))[0]
                    thresh_prob = zero_prob + \
                        (float(thresh_idxs.shape[0]) /
                         var_ser.shape[0])
                    thresh_prob_orig = thresh_prob
                    thresh_prob = zero_prob + (0.5 * (thresh_prob - zero_prob))

                    assert zero_prob <= thresh_prob, \
                        '\'zero_prob\' > \'thresh_prob\'!'

                    curr_py_zeros_dict[col] = zero_prob * 0.5
                    curr_py_dels_dict[col] = thresh_prob

                    var_ser_copy = var_ser.copy()
                    var_ser_copy[var_ser_copy <= self.var_ge_trs] = nan

                    probs_ser = var_ser_copy.rank() / \
                        (var_ser_copy.count() + 1.)
                    probs_ser = thresh_prob_orig + \
                        ((1.0 - thresh_prob_orig) * probs_ser)

                    probs_ser.iloc[zero_idxs] = (0.5 * zero_prob)

                    probs_ser.iloc[thresh_idxs] = thresh_prob

                    assert thresh_prob <= probs_ser.max(), \
                        '\'thresh_prob\' > \'probs_ser.max()\'!'
                else:
                    probs_ser = var_ser.rank() / (var_ser.count() + 1.)

                assert np_all(isfinite(probs_ser.values)), \
                    'NaNs in \'probs_ser\'!'

                probs_df[col] = probs_ser
                norms_df[col] = norm_ppf_py_arr(probs_ser.values)

                if (col == self.curr_infill_stn) and \
                        (self.infill_type == u'precipitation'):
                    py_del = thresh_prob
                    py_zero = zero_prob * 0.5

                curr_val_cdf_df = DataFrame(index=curr_var_df.index,
                                            columns=[_probs_str, _vals_str],
                                            dtype=float)
                curr_val_cdf_df[_probs_str] = probs_df[col].copy()
                curr_val_cdf_df[_vals_str] = var_ser.copy()

                curr_val_cdf_df.sort_values(by=_vals_str, inplace=True)

                curr_max_prob = curr_val_cdf_df[_probs_str].values.max()
                curr_min_prob = curr_val_cdf_df[_probs_str].values.min()

                curr_val_cdf_ftn = interp1d(curr_val_cdf_df[_vals_str].values,
                                            curr_val_cdf_df[_probs_str].values,
                                            bounds_error=False,
                                            fill_value=(curr_min_prob,
                                                        curr_max_prob))

                curr_val_cdf_ftns_dict[col] = curr_val_cdf_ftn

                if self.plot_diag_flag:
                    curr_norm_ppf_df = DataFrame(index=curr_var_df.index,
                                                 columns=[_probs_str,
                                                          _norm_str],
                                                 dtype=float)
                    curr_norm_ppf_df[_probs_str] = probs_df[col].copy()
                    curr_norm_ppf_df[_norm_str] = norms_df[col].copy()

                    curr_norm_ppf_df.sort_values(by=_probs_str, inplace=True)

                    # plot currently used stns CDFs
                    lg_1 = ax_1.scatter(curr_val_cdf_df[_vals_str].values,
                                        curr_val_cdf_df[_probs_str].values,
                                        label='CDF variable',
                                        alpha=0.5, color='r', s=0.5)

                    lg_2 = ax_2.scatter(curr_norm_ppf_df[_norm_str].values,
                                        curr_norm_ppf_df[_probs_str].values,
                                        label='CDF ui',
                                        alpha=0.9, color='b', s=0.5)

                    lgs = (lg_1, lg_2)
                    labs = [l.get_label() for l in lgs]
                    ax_1.legend(lgs, labs, loc=4, framealpha=0.5)

                    ax_1.grid()

                    ax_1.set_xlabel('variable x')
                    ax_2.set_xlabel('transformed variable x (ui)')
                    ax_1.set_ylabel('probability')
                    ax_1.set_ylim(0, 1)
                    ax_2.set_ylim(0, 1)

                    if self.infill_type == u'precipitation':
                        plt.suptitle(('Actual and normalized value CDFs '
                                      '(n=%d)\n stn: %s, date: %s\npy_zero: '
                                      '%0.2f, py_del: %0.2f') %
                                     (curr_val_cdf_df.shape[0],
                                      col,
                                      date_pref,
                                      zero_prob,
                                      thresh_prob))
                    else:
                        plt.suptitle(('Actual and normalized value CDFs '
                                      '(n=%d)\n stn: %s, date: %s' %
                                      (curr_val_cdf_df.shape[0],
                                       col,
                                       date_pref)))

                    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.8)
                    _ = 'CDF_%s_%s.%s' % (date_pref, col, self.out_fig_fmt)
                    out_cdf_fig_loc = os_join(self.stn_step_cdfs_dir, _)
                    plt.savefig(out_cdf_fig_loc,
                                dpi=self.out_fig_dpi,
                                bbox_inches='tight')
                    ax_1.cla()
                    ax_2.cla()
            return norms_df, py_del, py_zero

        def _get_full_corr(full_corrs_arr):
            temp_full_corrs_arr = full_corrs_arr.copy()
            temp_full_corrs_arr[temp_full_corrs_arr > self.max_corr] = 1.0
            temp_curr_stns = norms_df.columns.tolist()
            del_rows = []
            too_hi_corr_stns = []
            for row in range(temp_full_corrs_arr.shape[0]):
                for col in range(temp_full_corrs_arr.shape[1]):
                    if row > col:
                        if temp_full_corrs_arr[row, col] == 1.0:
                            too_hi_corr_stn = temp_curr_stns[row]
                            del_rows.append(row)

                            if too_hi_corr_stn not in too_hi_corr_stns:
                                too_hi_corr_stns.append(too_hi_corr_stn)

                            if too_hi_corr_stn not in too_hi_corr_stns_list:
                                too_hi_corr_stns_list.append(too_hi_corr_stn)

                            if too_hi_corr_stn in curr_val_cdf_ftns_dict:
                                del curr_val_cdf_ftns_dict[too_hi_corr_stn]

                            if too_hi_corr_stn in avail_cols_raw:
                                avail_cols_raw.remove(too_hi_corr_stn)

                            if too_hi_corr_stn in avail_cols_fin:
                                avail_cols_fin.remove(too_hi_corr_stn)

                            if too_hi_corr_stn in curr_var_df.columns:
                                curr_var_df.drop(labels=too_hi_corr_stn,
                                                 axis=1,
                                                 inplace=True)

            if len(del_rows) > 0:
                del_rows = unique(del_rows)
                temp_full_corrs_arr = delete(temp_full_corrs_arr,
                                             del_rows, axis=0)
                temp_full_corrs_arr = delete(temp_full_corrs_arr,
                                             del_rows, axis=1)

                if self.verbose:
                    print '\nWARNING: a correlation of almost equal',
                    print 'to one is encountered'
                    print 'Infill_stn:', self.curr_infill_stn
                    print 'Stations with correlation too high:\n', \
                        too_hi_corr_stns
                    print 'Infill_date:', infill_date
                    print 'These stations are added to the temporary high',
                    print ('correlations list and not used in '
                           'further processing')
            return temp_full_corrs_arr

        def _fill_u_t_and_get_val_cdf_ftn():
            u_t = full((curr_var_df.shape[1] - 1), nan)
            cur_vals = u_t.copy()

            for i, col in enumerate(curr_var_df.columns):
                # get u_t values or the interp ftns in case of infill_stn
                if i == 0:
                    val_cdf_ftn = curr_val_cdf_ftns_dict[col]
                    continue
                _curr_var_val = self.in_var_df.loc[infill_date, col]

                if self.infill_type == u'precipitation':
                    if _curr_var_val == self.var_le_trs:
                        values_arr = \
                            self.in_var_df.loc[infill_date,
                                               curr_var_df.columns[1:]
                                               ].dropna().values

                        if len(values_arr) > 0:
                            n_wet = (values_arr > self.var_le_trs).sum()
                            wt = n_wet / float(values_arr.shape[0])
                        else:
                            wt = 0.0

                        _ = curr_py_zeros_dict[col]
                        u_t[i - 1] = norm_ppf_py(_ * (1.0 + wt))

                    elif (_curr_var_val > self.var_le_trs) and \
                         (_curr_var_val <= self.var_ge_trs):
                        u_t[i - 1] = norm_ppf_py(curr_py_dels_dict[col])
                    else:
                        u_t[i - 1] = \
                            norm_ppf_py(
                                    curr_val_cdf_ftns_dict[col](_curr_var_val))
                else:
                    u_t[i - 1] = \
                        norm_ppf_py(
                                curr_val_cdf_ftns_dict[col](_curr_var_val))

                    cur_vals[i - 1] = _curr_var_val

            assert np_all(isfinite(u_t)), 'NaNs in \'u_t\'!'
            return cur_vals, u_t, val_cdf_ftn

        def _get_ppt_stuff():
            if curr_max_var_val > self.var_ge_trs:
                val_arr = val_cdf_ftn.x[val_cdf_ftn.x >= self.var_ge_trs]
                val_arr = append(linspace(self.var_le_trs,
                                          self.var_ge_trs,
                                          self.ge_le_trs_n,
                                          endpoint=False),
                                 val_arr)
            else:
                val_arr = linspace(curr_min_var_val,
                                   curr_max_var_val,
                                   self.ge_le_trs_n)

            assert val_arr.shape[0], '\'val_arr\' is empty!'

            gy_arr = full(val_arr.shape, nan)

            for i, val in enumerate(val_arr):
                if val > self.var_ge_trs:
                    _ = norm_ppf_py(val_cdf_ftn(val)) - mu_t
                    gy_arr[i] = norm_cdf_py(_ / sig_sq_t**0.5)
                elif (val > self.var_le_trs) and (val <= self.var_ge_trs):
                    _ = norm_ppf_py(py_del) - mu_t
                    gy_arr[i] = norm_cdf_py(_ / sig_sq_t**0.5)
                else:
                    values_arr = \
                        self.in_var_df.loc[infill_date,
                                           curr_var_df.columns[1:]
                                           ].dropna().values

                    if len(values_arr) > 0:
                        n_wet = (values_arr > self.var_le_trs).sum()
                        wt = n_wet / float(values_arr.shape[0])
                    else:
                        wt = 0.0

                    if py_zero:
                        _ = norm_ppf_py(py_zero * (1.0 + wt)) - mu_t
                        gy_arr[i] = norm_cdf_py(_ / sig_sq_t**0.5)
                    else:
                        gy_arr[i] = 0.0

                assert not isnan(gy_arr[i]), \
                    '\'gy\' is nan (val: %0.2e)!' % val

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_raw'] = gy_arr
                step_vars_dict['val_arr_raw'] = val_arr

            probs_idxs = gy_arr > self.adj_prob_bounds[0]
            probs_idxs = logical_and(probs_idxs,
                                     gy_arr < self.adj_prob_bounds[1])
            gy_arr = gy_arr[probs_idxs]
            val_arr = val_arr[probs_idxs]

            assert gy_arr.shape[0] > 0, 'Increase discretization!'
            assert gy_arr.shape[0] == val_arr.shape[0], \
                'unequal shapes of probs and vals!'

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_fin'] = gy_arr
                step_vars_dict['val_arr_fin'] = val_arr

            if len(gy_arr) == 1:
                # all probs are zero, hope so
                fin_val_ppf_ftn_adj = interp1d(linspace(0, 1.0, 10),
                                               [self.var_le_trs]*10,
                                               bounds_error=False,
                                               fill_value=(self.var_le_trs,
                                                           self.var_le_trs))
                fin_val_grad_ftn = interp1d((curr_min_var_val,
                                             curr_max_var_val),
                                            [0, 0],
                                            bounds_error=False,
                                            fill_value=(0, 0))
            else:
                fin_val_ppf_ftn = interp1d(gy_arr, val_arr,
                                           bounds_error=False,
                                           fill_value=(self.var_le_trs,
                                                       curr_max_var_val))

                curr_min_var_val_adj, curr_max_var_val_adj = \
                    fin_val_ppf_ftn([self.adj_prob_bounds[0],
                                     self.adj_prob_bounds[1]])

                # do the interpolation again with adjusted bounds
                if curr_max_var_val_adj > self.var_ge_trs:
                    adj_val_idxs = logical_and(val_arr >= self.var_ge_trs,
                                               val_arr <= curr_max_var_val_adj)
                    val_arr_adj = val_arr[adj_val_idxs]

                    if val_arr_adj.shape[0] < self.n_discret:
                        val_adj_interp = interp1d(range(0,
                                                        val_arr_adj.shape[0]),
                                                  val_arr_adj)
                        _interp_vals = linspace(0.0,
                                                val_arr_adj.shape[0] - 1,
                                                self.n_discret)
                        val_arr_adj = val_adj_interp(_interp_vals)

                    val_arr_adj = append(linspace(self.var_le_trs,
                                                  self.var_ge_trs,
                                                  self.ge_le_trs_n,
                                                  endpoint=False),
                                         val_arr_adj)
                else:
                    val_arr_adj = linspace(curr_min_var_val_adj,
                                           curr_max_var_val_adj,
                                           self.ge_le_trs_n)

                gy_arr_adj = full(val_arr_adj.shape, nan)
                pdf_arr_adj = gy_arr_adj.copy()

                for i, val_adj in enumerate(val_arr_adj):
                    if val_adj > self.var_ge_trs:
                        _ = (norm_ppf_py(val_cdf_ftn(val_adj)) - mu_t)
                        z_scor = _ / sig_sq_t**0.5
                        gy_arr_adj[i] = norm_cdf_py(z_scor)
                        pdf_arr_adj[i] = norm_pdf_py(z_scor)
                    elif (val_adj > self.var_le_trs) and \
                         (val_adj <= self.var_ge_trs):
                        _ = norm_ppf_py(py_del) - mu_t
                        z_scor = _ / sig_sq_t**0.5
                        gy_arr_adj[i] = norm_cdf_py(z_scor)
                        pdf_arr_adj[i] = norm_pdf_py(z_scor)
                    else:
                        values_arr = \
                            self.in_var_df.loc[infill_date,
                                               curr_var_df.columns[1:]
                                               ].dropna().values
                        # TODO: Check if this is true (the wt thing)
                        if len(values_arr) > 0:
                            n_wet = (values_arr > self.var_le_trs).sum()
                            wt = n_wet / float(values_arr.shape[0])
                        else:
                            wt = 0.0

                        if py_zero:
                            _ = norm_ppf_py(py_zero * (1.0 + wt)) - mu_t
                            z_scor = _ / sig_sq_t**0.5
                            gy_arr_adj[i] = norm_cdf_py(z_scor)
                            pdf_arr_adj[i] = norm_pdf_py(z_scor)
                        else:
                            gy_arr_adj[i] = pdf_arr_adj[i] = 0.0

                    assert not isnan(gy_arr_adj[i]), \
                        '\'gy\' is nan (val: %0.2e)!' % val_adj
                    assert not isnan(pdf_arr_adj[i]), \
                        '\'pdf\' is nan (val: %0.2e)!' % val_adj

                if self.save_step_vars_flag:
                    step_vars_dict['gy_arr_adj_raw'] = gy_arr_adj
                    step_vars_dict['val_arr_adj_raw'] = val_arr_adj
                    step_vars_dict['pdf_arr_adj_raw'] = pdf_arr_adj

                adj_probs_idxs = gy_arr_adj >= self.adj_prob_bounds[0]
                adj_probs_idxs = logical_and(
                                 adj_probs_idxs,
                                 gy_arr_adj <= self.adj_prob_bounds[1])

                gy_arr_adj = gy_arr_adj[adj_probs_idxs]
                pdf_arr_adj = pdf_arr_adj[adj_probs_idxs]
                val_arr_adj = val_arr_adj[adj_probs_idxs]

                assert gy_arr_adj.shape[0] > 0, 'Increase discretization!'
                assert gy_arr_adj.shape[0] == val_arr_adj.shape[0], \
                    'unequal shapes of probs and vals!'
                assert pdf_arr_adj.shape[0] == val_arr_adj.shape[0], \
                    'unequal shapes of densities and vals!'

                if self.save_step_vars_flag:
                    step_vars_dict['gy_arr_adj_fin'] = gy_arr_adj
                    step_vars_dict['val_arr_adj_fin'] = val_arr_adj
                    step_vars_dict['pdf_arr_adj_fin'] = pdf_arr_adj

                fin_val_ppf_ftn_adj = interp1d(gy_arr_adj,
                                               val_arr_adj,
                                               bounds_error=False,
                                               fill_value=(
                                                self.var_le_trs,
                                                curr_max_var_val_adj))
                fin_val_grad_ftn = interp1d(val_arr_adj,
                                            pdf_arr_adj,
                                            bounds_error=False,
                                            fill_value=(0, 0))
            return (fin_val_ppf_ftn_adj,
                    fin_val_grad_ftn,
                    gy_arr_adj,
                    val_arr_adj,
                    pdf_arr_adj)

        def _get_discharge_stuff():
            val_arr = val_cdf_ftn.x
            probs_arr = val_cdf_ftn.y
            gy_arr = full(probs_arr.shape, nan)
            for i, prob in enumerate(probs_arr):
                _ = norm_ppf_py(prob) - mu_t
                gy_arr[i] = norm_cdf_py(_ / sig_sq_t**0.5)
                assert not isnan(gy_arr[i]), \
                    '\'gy\' is nan (prob:%0.2e)!' % prob

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_raw'] = gy_arr

            # do the interpolation again with adjusted bounds
            adj_probs_idxs = gy_arr > self.adj_prob_bounds[0]
            adj_probs_idxs = logical_and(adj_probs_idxs,
                                         gy_arr < self.adj_prob_bounds[1])

            val_arr_adj = val_arr[adj_probs_idxs]
            probs_arr_adj = probs_arr[adj_probs_idxs]

            assert val_arr_adj.shape[0], '\'val_arr_adj\' is empty!'

            (curr_min_var_val_adj,
             curr_max_var_val_adj) = (val_arr_adj.min(),
                                      val_arr_adj.max())

            n_vals = where(adj_probs_idxs)[0].shape[0]
            if n_vals < self.n_discret:
                val_adj_interp = interp1d(range(0, val_arr_adj.shape[0]),
                                          val_arr_adj)
                prob_adj_interp = interp1d(range(0, val_arr_adj.shape[0]),
                                           probs_arr_adj)
                _interp_vals = linspace(0.0,
                                        val_arr_adj.shape[0] - 1,
                                        self.n_discret)
                val_arr_adj = val_adj_interp(_interp_vals)
                probs_arr_adj = prob_adj_interp(_interp_vals)

            gy_arr_adj = full(val_arr_adj.shape, nan)
            pdf_arr_adj = gy_arr_adj.copy()

            for i, adj_prob in enumerate(probs_arr_adj):
                z_scor = (norm_ppf_py(adj_prob) - mu_t) / sig_sq_t**0.5
                gy_arr_adj[i] = norm_cdf_py(z_scor)
                pdf_arr_adj[i] = norm_pdf_py(z_scor)

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_adj_fin'] = gy_arr_adj
                step_vars_dict['val_arr_adj_fin'] = val_arr_adj
                step_vars_dict['pdf_arr_adj_fin'] = pdf_arr_adj

            fin_val_ppf_ftn_adj = interp1d(gy_arr_adj,
                                           val_arr_adj,
                                           bounds_error=False,
                                           fill_value=(curr_min_var_val_adj,
                                                       curr_max_var_val_adj))
            fin_val_grad_ftn = interp1d(val_arr_adj,
                                        pdf_arr_adj,
                                        bounds_error=False,
                                        fill_value=(0, 0))
            return (fin_val_ppf_ftn_adj,
                    fin_val_grad_ftn,
                    gy_arr_adj,
                    val_arr_adj,
                    pdf_arr_adj)

        def _plot_step_cdf_pdf():
            # plot infill cdf
            plt.clf()
            plt.plot(val_arr_adj, gy_arr_adj)
            plt.scatter(conf_vals, conf_probs)
            if self.infill_type == u'precipitation':
                plt.title(('infill CDF\n stn: %s, date: %s\npy_zero: %0.2f, '
                           'py_del: %0.2f') %
                          (self.curr_infill_stn, date_pref, py_zero, py_del))
            else:
                plt.title('infill CDF\n stn: %s, date: %s' %
                          (self.curr_infill_stn, date_pref))
            plt.grid()

            plt_texts = []
            for i in range(conf_probs.shape[0]):
                plt_texts.append(plt.text(conf_vals[i],
                                          conf_probs[i],
                                          ('var_%0.2f: %0.2f' %
                                           (conf_probs[i], conf_vals[i])),
                                          va='top',
                                          ha='left'))

            adjust_text(plt_texts)

            _ = 'infill_CDF_%s_%s.%s' % (self.curr_infill_stn,
                                         date_pref,
                                         self.out_fig_fmt)
            out_val_cdf_loc = os_join(self.stn_infill_cdfs_dir, _)
            plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
            plt.savefig(out_val_cdf_loc,
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            plt.clf()

            # plot infill pdf
            plt.plot(val_arr_adj, pdf_arr_adj)
            plt.scatter(conf_vals, conf_grads)
            if self.infill_type == u'precipitation':
                plt.title(('infill PDF\n stn: %s, date: %s\npy_zero: %0.2f, '
                           'py_del: %0.2f') %
                          (self.curr_infill_stn, date_pref, py_zero, py_del))
            else:
                plt.title('infill PDF\n stn: %s, date: %s' %
                          (self.curr_infill_stn, date_pref))
            plt.grid()

            plt_texts = []
            for i in range(conf_probs.shape[0]):
                plt_texts.append(plt.text(conf_vals[i],
                                          conf_grads[i],
                                          ('var_%0.2f: %0.2e' %
                                           (conf_probs[i],
                                            conf_grads[i])),
                                          va='top',
                                          ha='left'))

            adjust_text(plt_texts)
            _ = 'infill_PDF_%s_%s.%s' % (self.curr_infill_stn,
                                         date_pref,
                                         self.out_fig_fmt)
            out_val_pdf_loc = os_join(self.stn_infill_pdfs_dir, _)
            plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
            plt.savefig(out_val_pdf_loc,
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            plt.clf()
            return

        def _plot_diag():
            # plot corrs
            tick_font_size = 3
            n_stns = full_corrs_arr.shape[0]

            corrs_ax = plt.subplot(111)
            corrs_ax.matshow(full_corrs_arr,
                             vmin=0,
                             vmax=1,
                             cmap=cmaps.Blues,
                             origin='lower')
            for s in zip(repeat(range(n_stns), n_stns),
                         tile(range(n_stns), n_stns)):
                corrs_ax.text(s[1],
                              s[0],
                              '%0.2f' % (full_corrs_arr[s[0], s[1]]),
                              va='center',
                              ha='center',
                              fontsize=tick_font_size)

            corrs_ax.set_xticks(range(0, n_stns))
            corrs_ax.set_xticklabels(curr_var_df.columns)
            corrs_ax.set_yticks(range(0, n_stns))
            corrs_ax.set_yticklabels(curr_var_df.columns)

            corrs_ax.spines['left'].set_position(('outward', 10))
            corrs_ax.spines['right'].set_position(('outward', 10))
            corrs_ax.spines['top'].set_position(('outward', 10))
            corrs_ax.spines['bottom'].set_position(('outward', 10))

            corrs_ax.tick_params(labelleft=True,
                                 labelbottom=True,
                                 labeltop=True,
                                 labelright=True)

            plt.setp(corrs_ax.get_xticklabels(),
                     size=tick_font_size,
                     rotation=45)
            plt.setp(corrs_ax.get_yticklabels(), size=tick_font_size)

            _ = 'stn_corrs_%s.%s' % (date_pref, self.out_fig_fmt)
            out_corrs_fig_loc = os_join(self.stn_step_corrs_dir, _)
            plt.savefig(out_corrs_fig_loc,
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            plt.clf()
            return

        ### INFILL
        out_conf_df = DataFrame(index=infill_dates,
                                columns=self.conf_ser.index,
                                dtype=float)
        out_add_info_df = DataFrame(index=infill_dates,
                                    dtype=float,
                                    columns=['infill_status',
                                             'n_neighbors_raw',
                                             'n_neighbors_fin',
                                             'act_val_prob'])

        out_add_info_df.loc[:, :] = [False, 0, 0, nan]

        _probs_str = 'probs'
        _norm_str = 'norm_vals'
        _vals_str = 'vals'

        bad_comb = False

        pre_avail_stns = [self.curr_infill_stn]
        too_hi_corr_stns_list = []

        if self.plot_diag_flag:
            ax_1 = plt.subplot(111)
            ax_2 = ax_1.twiny()

        for infill_date in infill_dates:
            date_pref = infill_date.strftime('%Y%m%d%H%M')

            if not isnan(self.in_var_df.loc[infill_date,
                                            self.curr_infill_stn]):
                if not self.compare_infill_flag:
                    continue

            if self.infill_type == u'precipitation':
                curr_vals = self.in_var_df.loc[infill_date,
                                               self.curr_nrst_stns
                                               ].dropna().values
                if np_all(curr_vals == self.var_le_trs):
                    out_conf_df.loc[infill_date] = self.var_le_trs
                    out_add_info_df.loc[infill_date, 0] = True
                    out_add_info_df.loc[infill_date, 1] = curr_vals.shape[0]
                    out_add_info_df.loc[infill_date, 2] = curr_vals.shape[0]
                    continue

            # see which stns are available at the given step
            avail_cols_raw = self.in_var_df.loc[infill_date,
                                                self.curr_nrst_stns
                                                ].dropna().index.tolist()

            if self.save_step_vars_flag:
                step_vars_dict = {}
                step_vars_dict['time'] = date_pref
                step_vars_curs = open(os_join(self.stn_step_vars_dir,
                                              date_pref + '.pkl'), 'wb')
                step_vars_dict['avail_cols_raw_bef_adj'] = avail_cols_raw

            corr_one_stns_remove_list = []
            for corr_one_stn in too_hi_corr_stns_list:
                if corr_one_stn not in corr_one_stns_remove_list:
                    corr_one_stns_remove_list.append(corr_one_stn)

            if len(corr_one_stns_remove_list) > 0:
                for corr_one_stn_neb in corr_one_stns_remove_list:
                    if corr_one_stn_neb in avail_cols_raw:
                        avail_cols_raw.remove(corr_one_stn_neb)

            out_add_info_df.loc[infill_date,
                                'n_neighbors_raw'] = len(avail_cols_raw)

            if self.save_step_vars_flag:
                step_vars_dict['avail_cols_raw_aft_adj'] = avail_cols_raw

            if len(avail_cols_raw) < self.n_nrst_stns_min:
                if (not self.force_infill_flag) or (len(avail_cols_raw) < 1):
                    bad_comb = True
                    pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw

                    if self.save_step_vars_flag:
                        dump(step_vars_dict, step_vars_curs, -1)
                        step_vars_curs.close()
                    continue

            if pre_avail_stns[1:] != avail_cols_raw:
                curr_val_cdf_ftns_dict = {}
                curr_py_zeros_dict = {}
                curr_py_dels_dict = {}

                _ = [self.curr_infill_stn] + avail_cols_raw
                curr_var_df = self.in_var_df[_].copy()
                best_stns = avail_cols_raw

                if self.use_best_stns_flag:
                    best_stns = _get_best_stns(avail_cols_raw)

                if self.save_step_vars_flag:
                    step_vars_dict['best_stns'] = list(best_stns)

                _ = [self.curr_infill_stn] + list(best_stns)
                curr_var_df = self.in_var_df[_].copy()

                if self.take_min_stns_flag:
                    curr_var_df = curr_var_df.iloc[:,
                                                   :self.n_nrst_stns_min + 1]

                curr_var_df.dropna(axis=0, inplace=True)
                if self.save_step_vars_flag:
                    step_vars_dict['cur_var_df_shape'] = curr_var_df.shape

                if curr_var_df.shape[0] < self.min_valid_vals:
                    bad_comb = True
                    pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw

                    if self.save_step_vars_flag:
                        dump(step_vars_dict, step_vars_curs, -1)
                        step_vars_curs.close()
                    continue

                try:
                    _ = curr_var_df.columns
                    avail_cols_fin = _.drop(self.curr_infill_stn).tolist()
                    out_add_info_df.loc[
                            infill_date,
                            'n_neighbors_fin'] = len(avail_cols_fin)
                except Exception as msg:
                    bad_comb = True
                    pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw
                    out_add_info_df.loc[infill_date, 'n_neighbors_fin'] = 0
                    print ('Apparently, infill stn is not there or it appears '
                           'more than once:', msg)
                    print '\'curr_var_df.columns:\'\n', curr_var_df.columns
                    if not self.dont_stop_flag:
                        raise RuntimeError
                    else:
                        continue

                if (curr_var_df.shape[1] - 1) < self.n_nrst_stns_min:
                    if (not self.force_infill_flag) or \
                       ((curr_var_df.shape[1] - 1) == 0):
                        bad_comb = True
                        pre_avail_stns = [self.curr_infill_stn] + \
                            avail_cols_raw
                        if self.save_step_vars_flag:
                            dump(step_vars_dict, step_vars_curs, -1)
                            step_vars_curs.close()
                        continue

                assert curr_var_df.shape[1] > 1, \
                    '\'curr_var_df\' has no neighboring stations in it!'
                assert curr_var_df.shape[0] > self.min_valid_vals, \
                    '\'curr_var_df\' has no records in it!'

                norms_df, py_del, py_zero = _create_cdfs_ftns_and_plots()

                full_corrs_arr = fill_correl_mat(norms_df.values)
                full_corrs_arr = _get_full_corr(full_corrs_arr)

                norm_cov_mat = full_corrs_arr[1:, 1:]
                cov_vec = full_corrs_arr[1:, 0]

                if cov_vec.shape[0] == 0:
                    print '\'cov_vec\' is empty on date:', infill_date
                    pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw
                    bad_comb = True
                    if self.save_step_vars_flag:
                        dump(step_vars_dict, step_vars_curs, -1)
                        step_vars_curs.close()
                    continue

                assert cov_vec.shape[0] > 0, '\'cov_vec\' is empty!'
                assert norm_cov_mat.shape[0] > 0, '\'norm_cov_mat\' is empty!'

                inv_norm_cov_mat = linalg.inv(norm_cov_mat)

                assert cov_vec.shape[0] == inv_norm_cov_mat.shape[0], \
                    'Incorrect deletion of vectors!'
                assert cov_vec.shape[0] == inv_norm_cov_mat.shape[1], \
                    'Incorrect deletion of vectors!'

                sig_sq_t = 1.0 - matmul(cov_vec.T,
                                        matmul(inv_norm_cov_mat, cov_vec))

                if sig_sq_t <= 0:
                    print ('Stn %s has invalid conditional variance' %
                           self.curr_infill_stn)
                    print '\'sig_sq_t (%0.6f)\' is less than zero!' % sig_sq_t
                    print '\'infill_date\':', infill_date
                    print '\'best_stns\': \'covariance\''
                    for bstn_cov in zip(best_stns.tolist(), cov_vec):
                        print '   %s: %0.7f' % (bstn_cov[0], bstn_cov[1])
                    print '\n'

                    pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw
                    bad_comb = True

                    if self.save_step_vars_flag:
                        dump(step_vars_dict, step_vars_curs, -1)
                        step_vars_curs.close()
                    continue

                curr_max_var_val = curr_var_df[self.curr_infill_stn].max()
                curr_min_var_val = curr_var_df[self.curr_infill_stn].min()

            elif not bad_comb:
                out_add_info_df.loc[infill_date,
                                    'n_neighbors_fin'] = len(avail_cols_fin)

            if bad_comb:
                pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw
                if self.save_step_vars_flag:
                    dump(step_vars_dict, step_vars_curs, -1)
                    step_vars_curs.close()
                continue

            bad_comb = False  # needed, don't remove

            if self.infill_type == u'precipitation':
                assert not isnan(py_zero), '\'py_zero\' is nan!'
                assert not isnan(py_del), '\'py_del\' is nan!'

            pre_avail_stns = [self.curr_infill_stn] + avail_cols_raw

            _ = curr_val_cdf_ftns_dict.keys()
            assert len(_) == len(avail_cols_fin) + 1, \
                ('\'curr_val_cdf_ftns_dict\' has incorrect number of keys!',
                 curr_val_cdf_ftns_dict.keys(), avail_cols_fin.tolist())

            cur_vals, u_t, val_cdf_ftn = _fill_u_t_and_get_val_cdf_ftn()
            mu_t = matmul(cov_vec.T, matmul(inv_norm_cov_mat, u_t))

            if self.save_step_vars_flag:
                step_vars_dict['best_stns'] = list(best_stns)
                step_vars_dict['u_t'] = u_t
                step_vars_dict['cur_vals'] = cur_vals
                step_vars_dict['act_val'] = \
                    self.in_var_df.loc[infill_date, self.curr_infill_stn]
                step_vars_dict['mu_t'] = mu_t
                step_vars_dict['sig_sq_t'] = sig_sq_t
                step_vars_dict['val_cdf_ftn'] = val_cdf_ftn
                step_vars_dict['cur_var_df_shape'] = curr_var_df.shape

                step_vars_dict['cov_vec_fin'] = cov_vec
                step_vars_dict['norm_cov_mat_fin'] = norm_cov_mat
                step_vars_dict['inv_norm_cov_mat'] = inv_norm_cov_mat

            if self.infill_type == u'precipitation':
                (fin_val_ppf_ftn_adj,
                 fin_val_grad_ftn,
                 gy_arr_adj,
                 val_arr_adj,
                 pdf_arr_adj) = _get_ppt_stuff()
            else:
                (fin_val_ppf_ftn_adj,
                 fin_val_grad_ftn,
                 gy_arr_adj,
                 val_arr_adj,
                 pdf_arr_adj) = _get_discharge_stuff()

            conf_probs = self.conf_ser.values
            if self.n_rand_infill_values:
                for rand_idx in range(self.conf_heads.shape[0],
                                      conf_probs.shape[0]):
                    conf_probs[rand_idx] = \
                        (self.adj_prob_bounds[0] +
                         ((self.adj_prob_bounds[1] -
                           self.adj_prob_bounds[0]) * random()))

            conf_vals = fin_val_ppf_ftn_adj(conf_probs)
            conf_grads = fin_val_grad_ftn(conf_vals)
            out_conf_df.loc[infill_date] = np_round(conf_vals, self.n_round)
            out_add_info_df.loc[infill_date, 'infill_status'] = True

            if self.save_step_vars_flag:
                step_vars_dict['conf_vals'] = conf_vals
                step_vars_dict['conf_grads'] = conf_grads
                dump(step_vars_dict, step_vars_curs, -1)
                step_vars_curs.close()

            if self.n_rand_infill_values:
                _ = ediff1d(conf_vals[:self.conf_heads.shape[0]])
                descend_idxs = where(_ < 0, 1, 0)
            else:
                descend_idxs = where(ediff1d(conf_vals) < 0, 1, 0)

            if np_any(descend_idxs):
                print '#'*30
                print (('Interpolated var_vals on %s at station: %s not in '
                        'ascending order!') %
                       (str(infill_date), str(self.curr_infill_stn)))
                print ('var_0.05',
                       'var_0.25',
                       'var_0.5',
                       'var_0.75',
                       'var_0.95:\n',
                       list(conf_vals))
                print 'gy:\n', list(gy_arr_adj)
                print 'theoretical_var_vals:\n', list(val_arr_adj)
                print '#'*30
                assert False, \
                    ('Interpolated var_vals on %s not in ascending order!' %
                     str(infill_date))

            if not isnan(self.in_var_df.loc[infill_date,
                                            self.curr_infill_stn]):
                fin_val_cdf_ftn_adj = interp1d(val_arr_adj,
                                               gy_arr_adj,
                                               bounds_error=False,
                                               fill_value=(
                                                self.adj_prob_bounds[+0],
                                                self.adj_prob_bounds[-1]))

                out_add_info_df.loc[infill_date, 'act_val_prob'] = \
                    fin_val_cdf_ftn_adj(
                            self.in_var_df.loc[infill_date,
                                               self.curr_infill_stn])

            if self.plot_step_cdf_pdf_flag:
                _plot_step_cdf_pdf()
            if self.plot_diag_flag:
                _plot_diag()

        plt.close()
        return (out_conf_df, out_add_info_df)

    def _plot_ecops(self, infill_stns):
        try:
            self.__plot_ecops(infill_stns)
        except:
            self._full_tb(exc_info())
        finally:
            plt.close()
        return

    def __plot_ecops(self, infill_stns):
        n_ticks = 6
        x_mesh, y_mesh = mgrid[0:self.cop_bins + 1, 0:self.cop_bins + 1]
        cop_ax_ticks = linspace(0, self.cop_bins, n_ticks)
        cop_ax_labs = np_round(linspace(0, 1., n_ticks, dtype='float'), 1)

        n_rows, n_cols = 1, 15 + 1

        plt.figure(figsize=(17, 6))
        ecop_raw_ax = plt.subplot2grid((n_rows, n_cols),
                                       (0, 0),
                                       rowspan=1,
                                       colspan=5)
        ecop_grid_ax = plt.subplot2grid((n_rows, n_cols),
                                        (0, 5),
                                        rowspan=1,
                                        colspan=5)
        gau_cop_ax = plt.subplot2grid((n_rows, n_cols),
                                      (0, 10),
                                      rowspan=1,
                                      colspan=5)
        leg_ax = plt.subplot2grid((n_rows, n_cols),
                                  (0, 15),
                                  rowspan=1,
                                  colspan=1)

        divider = make_axes_locatable(leg_ax)
        cax = divider.append_axes("left", size="100%", pad=0.05)

        if self.infill_type == 'discharge':
            dens_cnst = 0.5
        elif self.infill_type == 'precipitation':
            dens_cnst = 0.5

        for infill_stn in infill_stns:
            if (self.nrst_stns_type == 'rank') or \
               (self.nrst_stns_type == 'symm'):
                self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
            elif self.nrst_stns_type == 'dist':
                self.curr_nrst_stns = self.nrst_stns_dict[infill_stn]
            else:
                assert False, \
                    ('Incorrect \'nrst_stns_type\': %s!' %
                     str(self.nrst_stns_type))

            ser_i = self.in_var_df.loc[:, infill_stn].dropna().copy()
            ser_i_index = ser_i.index
            for other_stn in self.curr_nrst_stns:
                if infill_stn == other_stn:
                    continue

                ser_j = self.in_var_df.loc[:, other_stn].dropna().copy()
                index_ij = ser_i_index.intersection(ser_j.index)

                if index_ij.shape[0] < self.min_valid_vals:
                    continue

                new_ser_i = ser_i.loc[index_ij].copy()
                new_ser_j = ser_j.loc[index_ij].copy()

                ranks_i = new_ser_i.rank().values
                ranks_j = new_ser_j.rank().values

                prob_i = ranks_i / (new_ser_i.shape[0] + 1.)
                prob_j = ranks_j / (new_ser_j.shape[0] + 1.)

                # plot the empirical copula
                if prob_i.min() < 0 or prob_i.max() > 1:
                    raise Exception('\'prob_i\' values out of bounds!')
                if prob_j.min() < 0 or prob_j.max() > 1:
                    raise Exception('\'prob_j\' values out of bounds!')

                correl = get_corrcoeff(prob_i, prob_j)

                # random normal asymmetries
                asymms_1_list = []
                asymms_2_list = []
                for _ in range(self.n_norm_symm_flds):
                    as_1, as_2 = \
                        self._get_norm_rand_symms(correl)

                    asymms_1_list.append(as_1)
                    asymms_2_list.append(as_2)

                min_asymm_1 = min(asymms_1_list)
                max_asymm_1 = max(asymms_1_list)
                min_asymm_2 = min(asymms_2_list)
                max_asymm_2 = max(asymms_2_list)

                # Empirical copula - scatter
                ecop_raw_ax.scatter(prob_i,
                                    prob_j,
                                    alpha=0.9,
                                    color='b',
                                    s=0.5)
                ecop_raw_ax.set_xlabel('infill station: %s' % infill_stn)
                ecop_raw_ax.set_ylabel('other station: %s' % other_stn)
                ecop_raw_ax.set_xlim(0, 1)
                ecop_raw_ax.set_ylim(0, 1)
                ecop_raw_ax.grid()
                ecop_raw_ax.set_title('Empirical Copula - Scatter')

                # Empirical copula - gridded
                cop_dict = bi_var_copula(prob_i, prob_j, self.cop_bins)
                emp_dens_arr = cop_dict['emp_dens_arr']

                max_dens_idxs = where(emp_dens_arr == emp_dens_arr.max())
                max_dens_idx_i = max_dens_idxs[0][0]
                max_dens_idx_j = max_dens_idxs[1][0]
                emp_dens_arr_copy = emp_dens_arr.copy()
                emp_dens_arr_copy[max_dens_idx_i, max_dens_idx_j] = nan

                max_dens = nanmax(emp_dens_arr_copy) * dens_cnst
                ecop_grid_ax.pcolormesh(x_mesh,
                                        y_mesh,
                                        emp_dens_arr,
                                        cmap=cmaps.Blues,
                                        vmin=0,
                                        vmax=max_dens)
                ecop_grid_ax.set_xlabel('infill station: %s' % infill_stn)
                ecop_grid_ax.set_xticks(cop_ax_ticks)
                ecop_grid_ax.set_xticklabels(cop_ax_labs)
                ecop_grid_ax.set_yticks(cop_ax_ticks)
                ecop_grid_ax.set_yticklabels([])
                ecop_grid_ax.set_xlim(0, self.cop_bins)
                ecop_grid_ax.set_ylim(0, self.cop_bins)

                # get other copula params
                tau = tau_sample(prob_i, prob_j)

                emp_asymms = get_asymms_sample(prob_i, prob_j)
                emp_asymm_1, emp_asymm_2 = (emp_asymms['asymm_1'],
                                            emp_asymms['asymm_2'])

                asymm_1_str = 'within limits'
                if emp_asymm_1 < min_asymm_1:
                    asymm_1_str = 'too low'
                elif emp_asymm_1 > max_asymm_1:
                    asymm_1_str = 'too high'

                asymm_2_str = 'within limits'
                if emp_asymm_2 < min_asymm_2:
                    asymm_2_str = 'too low'
                elif emp_asymm_2 > max_asymm_2:
                    asymm_2_str = 'too high'

                emp_title_str = ''
                emp_title_str += 'Empirical copula - Gridded'
                emp_title_str += '\n(asymm_1: %1.1E, asymm_2: %1.1E)' % \
                                 (emp_asymm_1, emp_asymm_2)
                emp_title_str += '\n(asymm_1: %s, asymm_2: %s)' % \
                                 (asymm_1_str, asymm_2_str)

                ecop_grid_ax.set_title(emp_title_str)

                # Corresponding gaussian grid
                # TODO: adjust for precipitation case i.e. 0 and 1 ppt
                gau_cop_arr = bivar_gau_cop_arr(correl, self.cop_bins)
                _cb = gau_cop_ax.pcolormesh(x_mesh,
                                            y_mesh,
                                            gau_cop_arr,
                                            cmap=cmaps.Blues,
                                            vmin=0,
                                            vmax=max_dens)
                gau_cop_ax.set_xlabel('infill station: %s' % infill_stn)
                gau_cop_ax.set_xticks(cop_ax_ticks)
                gau_cop_ax.set_xticklabels(cop_ax_labs)
                gau_cop_ax.set_yticks(cop_ax_ticks)
                gau_cop_ax.set_yticklabels(cop_ax_labs)
                gau_cop_ax.tick_params(labelleft=False,
                                       labelbottom=True,
                                       labeltop=False,
                                       labelright=False)
                gau_cop_ax.set_xlim(0, self.cop_bins)
                gau_cop_ax.set_ylim(0, self.cop_bins)

                gau_title_str = ''
                gau_title_str += 'Gaussian copula'
                gau_title_str += (('\n(min asymm_1: %1.1E, '
                                   'max asymm_1: %1.1E)') % (min_asymm_1,
                                                             max_asymm_1))
                gau_title_str += (('\n(min asymm_2: %1.1E, '
                                   'max asymm_2: %1.1E)') % (min_asymm_2,
                                                             max_asymm_2))
                gau_cop_ax.set_title(gau_title_str)

                # legend
                leg_ax.set_axis_off()
                cb = plt.colorbar(_cb, cax=cax)
                cb.set_label('copula density')
                bounds = linspace(0, max_dens, 5)
                cb.set_ticks(bounds)
                cb.set_ticklabels(['%1.1E' % i_dens for i_dens in bounds])

                title_str = ''
                title_str += 'Copula densities of stations: %s and %s' % \
                             (infill_stn, other_stn)
                title_str += '\nn = %d, corr = %0.3f, bins = %d' % \
                             (prob_i.shape[0], correl, self.cop_bins)
                title_str += '\n(rho: %0.3f, tau: %0.3f)' % (correl, tau)
                plt.suptitle(title_str)

                plt.subplots_adjust(hspace=0.15, wspace=1.5, top=0.75)
                out_ecop_fig_loc = os_join(self.ecops_dir,
                                           ('ecop_%s_vs_%s.%s' %
                                            (infill_stn,
                                             other_stn,
                                             self.out_fig_fmt)))
                plt.savefig(out_ecop_fig_loc, dpi=self.out_fig_dpi)
                ecop_raw_ax.cla()
                ecop_grid_ax.cla()
                leg_ax.cla()
        return

    def _plot_infill_ser(self, args):
        '''
        Plot what the final series looks like
        '''
        (act_var, out_conf_df, out_infill_plot_loc) = args

        lw, alpha = 0.8, 0.7

        plt.figure(figsize=self.fig_size_long)
        infill_ax = plt.subplot(111)

        full_data_idxs = isnan(act_var)

        for _conf_head in out_conf_df.columns:
            if (not self.plot_rand_flag) and ('rand' in _conf_head):
                break

            conf_var_vals = where(full_data_idxs,
                                  out_conf_df[_conf_head].loc[
                                          self.infill_dates], act_var)
            infill_ax.plot(self.infill_dates,
                           conf_var_vals,
                           label=_conf_head,
                           alpha=alpha,
                           lw=lw,
                           ls='-',
                           marker='o',
                           ms=lw+0.5)

        infill_ax.plot(self.infill_dates,
                       act_var,
                       label='actual',
                       c='k',
                       ls='-',
                       marker='o',
                       alpha=1.0,
                       lw=lw+0.5,
                       ms=lw+1)

        infill_ax.set_xlabel('Time')
        infill_ax.set_ylabel('var_val')
        infill_ax.set_xlim(self.infill_dates[0], self. infill_dates[-1])
        infill_ax.set_title(('Infilled values for station: %s' %
                             str(self.curr_infill_stn)))
        plt.grid()
        plt.legend(framealpha=0.5, loc=0)
        plt.savefig(out_infill_plot_loc, dpi=self.out_fig_dpi)
        plt.close()
        return

    def _plot_compar_ser(self, args):
        '''
        1. Plot comparison between infilled (with CIs) and observed.
        2. Plot KS limits test
        3. Plot infill and observed historgrams comparison
        '''
        (act_var, out_conf_df, out_compar_plot_loc, out_add_info_df) = args

        lw, alpha = 0.8, 0.7

        if not self.n_rand_infill_values:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head].loc[
                                                          self.infill_dates
                                                          ].values))
        else:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head % 0].loc[
                                                          self.infill_dates
                                                          ].values))

        not_interp_data_idxs = logical_not(interp_data_idxs)

        plot_compare_cond = np_any(not_interp_data_idxs)
        plot_time_cdf_compare = True

        summ_df = DataFrame(index=[self.curr_infill_stn],
                            dtype=float,
                            columns=[self._compr_lab,
                                     self._ks_lims_lab,
                                     self._mean_obs_lab,
                                     self._mean_infill_lab,
                                     self._var_obs_lab,
                                     self._var_infill_lab,
                                     self._bias_lab,
                                     self._mae_lab,
                                     self._rmse_lab,
                                     self._nse_lab,
                                     self._ln_nse_lab,
                                     self._kge_lab,
                                     self._pcorr_lab,
                                     self._scorr_lab])

        if plot_compare_cond:
            # compare the observed and infill bounds and plot
            orig_vals = act_var[not_interp_data_idxs]

            if not self.n_rand_infill_values:
                infill_vals = out_conf_df[self.fin_conf_head]
            else:
                infill_vals = out_conf_df[self.fin_conf_head % 0]

            infill_vals = infill_vals.loc[self.infill_dates].values
            infill_vals = infill_vals[not_interp_data_idxs]

            n_vals = orig_vals.shape[0]

            diff = orig_vals - infill_vals

            bias = round(np_sum(diff) / n_vals, self.n_round)
            mae = round(np_sum(np_abs(diff)) / n_vals, self.n_round)
            rmse = round((np_sum(diff**2) / n_vals)**0.5, self.n_round)

            orig_probs = rankdata(orig_vals) / (n_vals + 1.)
            orig_probs_sort_idxs = argsort(orig_probs)
            orig_probs = orig_probs[orig_probs_sort_idxs]

            infill_probs = rankdata(infill_vals) / (n_vals + 1.)
            infill_probs = infill_probs[orig_probs_sort_idxs]

            nse = round(get_ns_py(orig_vals, infill_vals, 0),
                        self.n_round)
            ln_nse = round(get_ln_ns_py(orig_vals, infill_vals, 0),
                           self.n_round)
            kge = round(get_kge_py(orig_vals, infill_vals, 0),
                        self.n_round)

            correl_pe = round(get_corrcoeff(orig_vals, infill_vals),
                              self.n_round)
            correl_sp = round(get_corrcoeff(orig_probs, infill_probs),
                              self.n_round)

            obs_mean = round(orig_vals.mean(), self.n_round)
            infill_mean = round(infill_vals.mean(), self.n_round)
            obs_var = round(orig_vals.var(), self.n_round)
            infill_var = round(infill_vals.var(), self.n_round)

            summ_df.loc[self.curr_infill_stn,
                        [self._compr_lab,
                         self._bias_lab,
                         self._mae_lab,
                         self._rmse_lab,
                         self._nse_lab,
                         self._ln_nse_lab,
                         self._kge_lab,
                         self._pcorr_lab,
                         self._scorr_lab]] = \
                [n_vals,
                 bias,
                 mae,
                 rmse,
                 nse,
                 ln_nse,
                 kge,
                 correl_pe,
                 correl_sp]

            summ_df.loc[self.curr_infill_stn,
                        [self._mean_obs_lab,
                         self._mean_infill_lab,
                         self._var_obs_lab,
                         self._var_infill_lab]] = [obs_mean,
                                                   infill_mean,
                                                   obs_var,
                                                   infill_var]

            plt.figure(figsize=self.fig_size_long)
            infill_ax = plt.subplot(111)

            for _conf_head in out_conf_df.columns:
                conf_var_vals = where(interp_data_idxs, nan,
                                      out_conf_df[_conf_head].loc[
                                              self.infill_dates])

                infill_ax.plot(self.infill_dates,
                               conf_var_vals,
                               label=_conf_head,
                               alpha=alpha,
                               lw=lw,
                               ls='-',
                               marker='o',
                               ms=lw+0.5)

            infill_ax.plot(self.infill_dates,
                           where(interp_data_idxs, nan, act_var),
                           label='actual',
                           c='k',
                           ls='-',
                           marker='o',
                           alpha=1.0,
                           lw=lw+0.5,
                           ms=lw+1)

            infill_ax.set_xlabel('Time')
            infill_ax.set_ylabel('var_val')
            title_str = (('Observed and infill confidence interval '
                          'comparison for station: %s (%d values)') %
                         (self.curr_infill_stn, n_vals))
            title_str += (('\n(Bias: %0.2f, Mean absoulte Error: %0.2f, '
                           'Root mean squared error: %0.2f)') %
                          (bias, mae, rmse))
            title_str += (('\n(NSE: %0.2f, Ln-NSE: %0.2f, KGE: %0.2f, ') %
                          (nse, ln_nse, kge))
            title_str += (('Pearson correlation: %0.2f, Spearman correlation: '
                           '%0.2f)') % (correl_pe, correl_sp))
            plt.suptitle(title_str)
            infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
            plt.grid()
            plt.legend(framealpha=0.5, loc=0)
            plt.savefig(out_compar_plot_loc, dpi=self.out_fig_dpi)
            plt.close()

        if plot_compare_cond and plot_time_cdf_compare:
            # plot the infilled cdf values against observed w.r.t time
            plt.figure(figsize=(6, 5.5))
            infill_ax = plt.subplot(111)
            infill_ax.plot(infill_probs,
                           orig_probs,
                           alpha=alpha,
                           lw=lw,
                           ls='-',
                           marker='o',
                           ms=lw)
            infill_ax.plot(orig_probs,
                           orig_probs,
                           alpha=0.25,
                           c='k',
                           lw=lw+6,
                           ls='-')

            infill_ax.set_xlabel('Infilled Probability')
            infill_ax.set_ylabel('Observed Probability')

            infill_ax.set_xlim(-0.05, 1.05)
            infill_ax.set_ylim(-0.05, 1.05)

            title_str = (('Observed and infilled probability comparison '
                          'for each \ninfilled value for station: '
                          '%s') % self.curr_infill_stn)
            plt.suptitle(title_str)

            plt.grid()
            _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
            out_freq_compare_loc = _ + '_time_cdf.' + self.out_fig_fmt
            plt.savefig(out_freq_compare_loc, dpi=self.out_fig_dpi)
            plt.close()

            # plot distribution of probabilities of observed values in
            # the cdf for a given time step
            obs_probs_ser = out_add_info_df['act_val_prob'].copy()
            obs_probs_ser.dropna(inplace=True)

            obs_probs = obs_probs_ser.values
            obs_arg_sort = obs_probs.argsort()
            obs_probs = obs_probs[obs_arg_sort]

            obs_probs_probs = obs_probs_ser.rank() / \
                (obs_probs_ser.shape[0] + 1.)
            obs_probs_probs = obs_probs_probs.values
            obs_probs_probs = obs_probs_probs[obs_arg_sort]

            ks_d_stat = ((-0.5 * mlog(self.ks_alpha * 0.5)) /
                         obs_probs.shape[0])**0.5

            ks_fl = obs_probs_probs - ks_d_stat
            ks_fl[ks_fl < 0] = 0

            ks_fu = obs_probs_probs + ks_d_stat
            ks_fu[ks_fu > 1] = 1

            plt.figure(figsize=(6, 5.5))
            obs_probs_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            obs_probs_hist_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

            obs_probs_ax.plot(obs_probs,
                              obs_probs_probs,
                              alpha=alpha,
                              lw=lw,
                              ls='-',
                              marker='o',
                              ms=lw)
            obs_probs_ax.plot(obs_probs_probs,
                              ks_fl,
                              alpha=alpha,
                              c='k',
                              lw=lw,
                              ls='--',
                              ms=lw)
            obs_probs_ax.plot(obs_probs_probs,
                              ks_fu,
                              alpha=alpha,
                              c='k',
                              lw=lw,
                              ls='--',
                              ms=lw)

            obs_probs_ax.set_xticklabels([])
            obs_probs_ax.set_xlim(-0.05, 1.05)
            obs_probs_ax.set_ylim(-0.05, 1.05)
            obs_probs_ax.set_ylabel('Theoretical Probability')
            obs_probs_ax.grid()
            obs_probs_ax.get_xaxis().set_tick_params(width=0)

            obs_probs_hist_ax.hist(obs_probs,
                                   bins=20,
                                   alpha=alpha,
                                   range=(0.0, 1.0))
            obs_probs_hist_ax.set_xlim(-0.05, 1.05)
            obs_probs_hist_ax.set_xlabel('Observed-infilled Probability')
            obs_probs_hist_ax.set_ylabel('Frequency')
            obs_probs_hist_ax.grid()

            n_ks_fu_g = where(obs_probs < ks_fl)[0].shape[0]
            n_ks_fl_l = where(obs_probs > ks_fu)[0].shape[0]

            vals_wtn_rng = 100.0 * (n_vals - n_ks_fl_l - n_ks_fu_g) / \
                float(n_vals)

            summ_df.loc[self.curr_infill_stn,
                        self._ks_lims_lab] = round(vals_wtn_rng, 2)

            title_str = (('Observed values\' infilled probability for '
                          'station: %s') % self.curr_infill_stn)
            title_str += (('\n%0.2f%% values within %0.0f%% KS-limits') %
                          (vals_wtn_rng, 100 * (1.0 - self.ks_alpha)))
            title_str += (('\nOut of %d, %d values below and %d above '
                           'limits') % (n_vals, n_ks_fl_l, n_ks_fu_g))
            plt.suptitle(title_str)

            _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
            out_freq_compare_loc = _ + '_obs_probs_cdf.' + self.out_fig_fmt
            plt.savefig(out_freq_compare_loc, dpi=self.out_fig_dpi)
            plt.close()

            # plot the CDFs of infilled and original data
            orig_sorted_val_idxs = argsort(orig_vals)
            infill_sorted_val_idxs = argsort(infill_vals)

            orig_vals = orig_vals[orig_sorted_val_idxs]
            infill_vals = infill_vals[infill_sorted_val_idxs]

            orig_probs = rankdata(orig_vals) / (n_vals + 1.)
            infill_probs = rankdata(infill_vals) / (n_vals + 1.)

            plt.figure(figsize=(6, 5.5))
            infill_ax = plt.subplot(111)

            _min_var = max(orig_vals.min(), infill_vals.min())
            _max_var = max(orig_vals.max(), infill_vals.max())

            infill_ax.hist(orig_vals,
                           bins=20,
                           range=(_min_var, _max_var),
                           alpha=0.5,
                           label='observed')
            infill_ax.hist(infill_vals,
                           bins=20,
                           range=(_min_var, _max_var),
                           rwidth=0.8,
                           alpha=0.5,
                           label='infilled')

            infill_ax.set_xlabel('Variable')
            infill_ax.set_ylabel('Frequency')
            title_str = (('Observed and infilled histogram comparison '
                          '\nfor station: %s (%d values)') %
                         (self.curr_infill_stn, n_vals))

            plt.suptitle(title_str)
            plt.grid()
            plt.legend()
            _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
            out_cdf_compare_loc = _ + '_hists.' + self.out_fig_fmt
            plt.savefig(out_cdf_compare_loc, dpi=self.out_fig_dpi)
            plt.close()
        return summ_df

    def _plot_flag_susp_ser(self, args):
        '''
        Plot the flags for values that are out of the normal copula CI bounds
        '''
        (act_var, out_conf_df, out_flag_susp_loc) = args

        if not self.n_rand_infill_values:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head].loc[
                                                          self.infill_dates
                                                          ].values))
        else:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head % 0].loc[
                                                          self.infill_dates
                                                          ].values))

        _conf_head_list = []
        for _conf_head in out_conf_df.columns:
            if self.conf_ser[_conf_head] in self.flag_probs:
                _conf_head_list.append(_conf_head)

        conf_var_vals_lo = \
            out_conf_df[_conf_head_list[0]].loc[self.infill_dates].values
        conf_var_vals_hi = \
            out_conf_df[_conf_head_list[1]].loc[self.infill_dates].values

        conf_var_vals_lo[isnan(conf_var_vals_lo)] = -inf
        conf_var_vals_hi[isnan(conf_var_vals_hi)] = +inf

        not_interp_data_idxs = logical_not(interp_data_idxs)

        act_var_lo = act_var.copy()
        act_var_hi = act_var.copy()

        act_var_lo[isnan(act_var)] = +inf
        act_var_hi[isnan(act_var)] = -inf

        flag_arr = full(act_var.shape[0], nan)
        conf_var_idxs_lo = where(not_interp_data_idxs,
                                 act_var_lo < conf_var_vals_lo,
                                 False)
        conf_var_idxs_hi = where(not_interp_data_idxs,
                                 act_var_hi > conf_var_vals_hi,
                                 False)
        conf_var_idxs_wi_1 = where(not_interp_data_idxs,
                                   act_var_hi >= conf_var_vals_lo,
                                   False)
        conf_var_idxs_wi_2 = where(not_interp_data_idxs,
                                   act_var_lo <= conf_var_vals_hi,
                                   False)
        conf_var_idxs_wi = logical_and(conf_var_idxs_wi_1,
                                       conf_var_idxs_wi_2)

        flag_arr[conf_var_idxs_lo] = -1
        flag_arr[conf_var_idxs_hi] = +1
        flag_arr[conf_var_idxs_wi] = +0

        flag_arr[interp_data_idxs] = nan  # just in case

        _flag_ser = self.flag_df[self.curr_infill_stn].copy()
        _flag_ser[:] = flag_arr

        n_below_lower_lims = where(conf_var_idxs_lo)[0].shape[0]
        n_above_upper_lims = where(conf_var_idxs_hi)[0].shape[0]
        n_within_lims = where(conf_var_idxs_wi)[0].shape[0]

        summ_df = DataFrame(index=[self.curr_infill_stn],
                            columns=[self._flagged_lab],
                            dtype=float)

        n_out_bds = n_below_lower_lims + n_above_upper_lims
        n_tot = n_out_bds + n_within_lims

        if n_tot:
            summ_df.iloc[0, 0] = \
                 100 * ((n_out_bds + 0.0) / (n_tot))
            summ_df.iloc[0, 0] = round(summ_df.iloc[0, 0], self.n_round)

        flag_str = '(steps below limits: %d, ' % n_below_lower_lims
        flag_str += 'steps within limits: %d, ' % n_within_lims
        flag_str += 'steps above limits: %d)' % n_above_upper_lims

        lw, alpha = 0.8, 0.7

        plt.figure(figsize=self.fig_size_long)
        infill_ax = plt.subplot(111)
        infill_ax.plot(self.infill_dates,
                       flag_arr,
                       alpha=alpha,
                       lw=lw+0.5,
                       ls='-')

        infill_ax.set_xlabel('Time')
        infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
        infill_ax.set_ylabel('Flag')
        infill_ax.set_yticks([-1, 0, 1])
        infill_ax.set_ylim(-2, 2)
        _y_ticks = ['below_%0.3fP' % self.flag_probs[0],
                    'within\n%0.3fP_&_%0.3fP' % (self.flag_probs[0],
                                                 self.flag_probs[1]),
                    'above_%0.3fP' % self.flag_probs[1]]
        infill_ax.set_yticklabels(_y_ticks)

        plt.suptitle(('Data quality flags for station: %s\n' %
                      self.curr_infill_stn + flag_str))
        plt.grid()
        plt.savefig(out_flag_susp_loc, dpi=self.out_fig_dpi)
        plt.close()
        return summ_df, _flag_ser


if __name__ == '__main__':
    pass
