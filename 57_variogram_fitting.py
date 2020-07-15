import timeit
import geostatistics_mixed_vg as gs
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
#from return_covmod import return_fitted_covmod, make_flags
#from covariancefunction import Covariogram


def get_cell_size(x, y):
    x, y = pd.np.ravel(x), pd.np.ravel(y)
    cell_size_adj = 0.99
    xx1, xx2 = pd.np.meshgrid(x, x)
    yy1, yy2 = pd.np.meshgrid(y, y)
    dists = pd.np.sqrt((xx2-xx1)**2 + (yy2-yy1)**2)
    cell_size = cell_size_adj*pd.np.sqrt(pow(pd.np.unique(dists)[1], 2) / 2)
    return float(cell_size)


def get_sill_and_range(model):
    models = model.split('+')
    models = pd.np.array(models)
    Sill = 0.0
    # go through models:
    for submodel in models:
        submodel = submodel.strip()
        Sill  += float(submodel.split('(')[0].strip()[:-3].strip())
        Range = submodel.split('(')[1].split(')')[0]
    return [float(Sill), float(Range)]


def make_variogram(in_vals_df, out_vars_df_cols, in_coords_df, out_figs_dir,
                   out_q, nk=0.75, mdr=0.5, h_typ='var', perm_r_list=[1, 2],
                   fit_vgs=['all'], fil_nug_vg='Sph', n_best=3,
                   evg_name='classic', ngp=5, min_valid_stns=20):
    plt.ioff()
#    fig_size = (20, 10)

    vg_names_df = pd.DataFrame(index=in_vals_df.index, columns=out_vars_df_cols)
    vg_sills_df = pd.DataFrame(index=in_vals_df.index, columns=out_vars_df_cols)

    for date in in_vals_df.index:
        in_vals_ser = in_vals_df.loc[date].copy()
        in_vals_ser.dropna(inplace=True)

        date_str ='%0.4d-%0.2d-%0.2d'  % (date.year, date.month, date.day)

        aval_stns = in_vals_ser.index.intersection(in_coords_df.dropna().index)

        if aval_stns.shape[0] < min_valid_stns:
            continue

        x, y = in_coords_df.loc[aval_stns]['X'].values, in_coords_df.loc[aval_stns]['Y'].values
        z = in_vals_ser.loc[aval_stns].values

        variogram = gs.Variogram(x=x,
                                 y=y,
                                 z=z,
                                 mdr=mdr,
                                 typ=h_typ,
                                 perm_r_list=perm_r_list,
                                 fil_nug_vg=fil_nug_vg,
                                 ld=None,
                                 uh=None,
                                 h_itrs=100,
                                 opt_meth='L-BFGS-B',
                                 opt_iters=1000,
                                 fit_vgs=fit_vgs,
                                 n_best=n_best,
                                 evg_name=evg_name,
                                 use_wts=True,
                                 ngp=ngp,
                                 fit_thresh=0.01)

        #==============================================================================
        # Mix vg strt
        #==============================================================================

        variogram.call_vg()

#            print 'Total fitted variograms: %d' % len(variogram.best_vg_names)
        fit_vg_list = variogram.vg_str_list
        fit_vgs_no = len(fit_vg_list) - 1

        for i, vg_str in enumerate(fit_vg_list):
            vg_names_df.loc[date][fit_vgs_no - i] = vg_str
            vg_sills_df.loc[date][fit_vgs_no - i] = variogram.vg_variance_list[i]

        evg = variogram.vg_vg_arr
        h_arr = variogram.vg_h_arr
        vg_fit = variogram.vg_fit

#        plt.figure(figsize=fig_size)
        plt.figure()
        plt.plot(h_arr, evg, 'bo', alpha=0.3)

        vg_names = variogram.best_vg_names

        for m in range(len(vg_names)):
            plt.plot(vg_fit[m][:, 0], vg_fit[m][:, 1], c=pd.np.random.rand(3,), linewidth=4, zorder=m, label=fit_vg_list[m])

        plt.grid()
        plt.xlabel('Distance (cells)')
        plt.ylabel('Variogram')
        plt.title('Event date: %s' % (date_str), fontdict={'fontsize':15})

        plt.legend(loc=4, framealpha=0.5)

        plt.savefig(out_figs_dir + date_str + '.png')
        plt.close()

    out_q.put((vg_names_df, vg_sills_df))

#==============================================================================
# Main ftn
#==============================================================================
if __name__ == '__main__':
    import os
    print '\a\a\a\a Start \a\a\a\a'
    start = timeit.default_timer() # to get the runtime of the program

#==============================================================================
## TEMPERATURE
#    main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep'
#    in_vals_df_loc = os.path.join(
#             r'santa_infill_temperature_Fabian_May2017_07_monthly_resamp',
#             r'infilled_var_df_infill_stns.csv')
#    in_stn_coords_df_loc = r'santa_temp_coords.csv'
#
#    out_dir = r'santa_temperature_kriging_03'
#    suff = 'temp'
#==============================================================================


#==============================================================================
## PRECIPITATION
#    main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep'
#    in_vals_df_loc = os.path.join(
#             r'santa_infill_precipitation_Fabian_May2017_15_monthly_resamp',
#             r'infilled_var_df_infill_stns.csv')
#    in_stn_coords_df_loc = r'santa_ppt_coords.csv'
#
#    out_dir = r'santa_precipitation_kriging_01'
#    suff = 'ppt'
#==============================================================================

#==============================================================================
# POTENTIAL EVAPOTRANSPIRATION
    main_dir = r'P:\Synchronize\IWS\2015_Water_Balance_Peru\04_hydrological_data_prep'
    in_vals_df_loc = r'santa_pet_may17.csv'
    in_stn_coords_df_loc = r'santa_temp_coords.csv'

    out_dir = r'santa_pet_kriging_03'
    suff = 'pet'
#==============================================================================

    strt_date = '1963-01-01'
    end_date = '2015-12-31'
    min_valid_stns = 20

#==============================================================================
#     figures and out df directories are defined below
#==============================================================================

    drop_stns = []
    nk = 1
    mdr = 0.9
    h_typ = 'var'
    perm_r_list = [2]
    fit_vgs = ['Sph', 'Exp', 'Gau', 'Lin', 'Hol']
    fil_nug_vg = 'Nug'
    n_best = 2
    evg_name = 'robust'
    ngp = 5

    tot_procs = 3

    test_vg_ftn = False

    sep = ';'

    os.chdir(main_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    in_vals_df = pd.read_csv(in_vals_df_loc, sep=sep, index_col=0, encoding='utf-8')
    in_vals_df.index = pd.to_datetime(in_vals_df.index, format='%Y-%m-%d')
    in_vals_df = in_vals_df.loc[strt_date:end_date, :]
    in_vals_df.drop(labels=drop_stns, axis=1, inplace=True)
    in_vals_df.dropna(how='all', axis=0, inplace=True)

    in_coords_df = pd.read_csv(in_stn_coords_df_loc, sep=sep, index_col=0, encoding='utf-8')
    in_coords_df.index = map(unicode, in_coords_df.index)
    in_coords_df.drop(labels=drop_stns, axis=0, inplace=True)

    sub_name_str = '__nk_%0.3f__evg_name_%s__ngp_%d__h_typ_%s' % (nk, evg_name, ngp, h_typ)
    out_figs_dir = os.path.join(out_dir, r'%s_variogram_figs%s/' % (suff, sub_name_str))
    out_vars_df_loc = os.path.join(out_dir, r'%s_fitted_variograms%s.csv' % (suff, sub_name_str))
    out_sills_df_loc = os.path.join(out_dir, r'%s_sills%s.csv' % (suff, sub_name_str))

    if not os.path.exists(out_figs_dir):
        os.mkdir(out_figs_dir)

    fit_vg_cols = []
    fit_vg_sills = []
    for i in range(n_best):
        fit_vg_cols.append('best_fit_vg_no_%0.2d' % (i+1))
        fit_vg_sills.append('best_fit_sill_no_%0.2d' % (i+1))

    out_vars_df = pd.DataFrame(index=in_vals_df.index, columns=fit_vg_cols)
    out_sills_df = pd.DataFrame(index=in_vals_df.index, columns=fit_vg_sills)

    x, y = in_coords_df['X'].values, in_coords_df['Y'].values
#    cell_size = get_cell_size(x, y)


#==============================================================================
# multiproc start
#==============================================================================
    out_q = mp.Queue()
    processes = []

    tot_recs = in_vals_df.shape[0]
    print '\nTotal records to process: %d' % tot_recs

#==============================================================================
# Start make_variogram function test
#==============================================================================
    if test_vg_ftn:
        make_variogram(in_vals_df[:5],
                       fit_vg_cols,
                       in_coords_df,
                       out_figs_dir,
                       out_q,
                       nk,
                       mdr,
                       h_typ,
                       perm_r_list,
                       fit_vgs,
                       fil_nug_vg,
                       n_best,
                       evg_name,
                       ngp,
                       min_valid_stns)
#==============================================================================
#  End make_variogram function test
#==============================================================================
    else:
        idxs = pd.np.linspace(0, tot_recs, tot_procs + 1, endpoint=True, dtype='int64')
        if idxs.shape[0] == 1:
            idxs = pd.np.concatenate((pd.np.array([0]), idxs))

        for idx in range(tot_procs):
            sub_df =  in_vals_df.iloc[idxs[idx]:idxs[idx+1]]

            p = mp.Process(target=make_variogram, args=(sub_df,
                                                        fit_vg_cols,
                                                        in_coords_df,
                                                        out_figs_dir,
                                                        out_q,
                                                        nk,
                                                        mdr,
                                                        h_typ,
                                                        perm_r_list,
                                                        fit_vgs,
                                                        fil_nug_vg,
                                                        n_best,
                                                        evg_name,
                                                        ngp,
                                                        min_valid_stns))
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            sub_out_vars_df, sub_out_sills_df = out_q.get()
            out_vars_df.loc[sub_out_vars_df.index] = sub_out_vars_df
            out_sills_df.loc[sub_out_sills_df.index] = sub_out_sills_df

        for p in processes:
            p.join()

    #==============================================================================
    # multiproc end
    #==============================================================================

        out_vars_df.to_csv(out_vars_df_loc, sep=sep)
        out_sills_df.to_csv(out_sills_df_loc, sep=sep)

    stop = timeit.default_timer() # Ending time
    print '\n\a\a\a Done with everything. Total run time was about %s seconds \a\a\a' % (round(stop-start, 4))