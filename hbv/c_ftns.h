#include <math.h>
#include <stdio.h>


double get_ns_c(const double *x_arr, double *y_arr,
                const unsigned long *size,
                const double *demr,
                const unsigned long *off_idx) {

    unsigned long i;
    double numr = 0, curr_diff;

    for (i=(*off_idx); i<*size; ++i) {
        curr_diff = (x_arr[i] - y_arr[i]);
        numr += (curr_diff * curr_diff);
    }

    return (1.0 - (numr / *demr));
}


double get_ln_ns_c(const double *x_arr, double *y_arr,
                   const unsigned long *size,
                   const double *ln_demr,
                   const unsigned long *off_idx) {

    unsigned long i;
    double ln_numr = 0, curr_diff;

    for (i=(*off_idx); i<*size; ++i) {
        curr_diff = (log(x_arr[i]) - log(y_arr[i]));
        ln_numr += (curr_diff * curr_diff);
    }

    return (1.0 - (ln_numr / *ln_demr));
}


double get_mean_c(const double *in_arr,
                  const unsigned long *size,
                  const unsigned long *off_idx) {

    double _sum = 0.0;
    unsigned long i;

    for (i=*off_idx; i<*size; ++i) {
        _sum += in_arr[i];
    }
    return _sum / (*size - *off_idx);
}


double get_var_c(const double *in_arr_mean,
                 const double *in_arr,
                 const unsigned long *size,
                 const unsigned long *off_idx) {

    double _sum = 0;
    unsigned long i;

    for(i=*off_idx; i<*size; ++i) {
        _sum += pow((in_arr[i] - *in_arr_mean), 2);
    }
    return _sum / (*size - *off_idx);
}


double get_covar_c(const double *in_arr_1_mean,
                   const double *in_arr_2_mean,
                   const double *in_arr_1,
                   const double *in_arr_2,
                   const unsigned long *size,
                   const unsigned long *off_idx) {

    double _sum = 0;
    unsigned long i;

    for(i=*off_idx; i<*size; ++i){
        _sum += ((in_arr_1[i] - *in_arr_1_mean) * \
                 (in_arr_2[i] - *in_arr_2_mean));
    }
    return _sum / (*size - *off_idx);
}


double get_corr_c(const double *in_arr_1_std_dev,
                  const double *in_arr_2_std_dev,
                  const double *arrs_covar) {

    return *arrs_covar / (*in_arr_1_std_dev * *in_arr_2_std_dev);
}


double get_kge_c(const double *act_arr,
                 const double *sim_arr,
                 const double *act_mean,
                 const double *act_std_dev,
                 const unsigned long *size,
                 const unsigned long *off_idx) {

    double sim_mean, sim_std_dev, covar;
    double correl, b, g, kge;

    sim_mean = get_mean_c(sim_arr, size, off_idx);
    sim_std_dev = pow(get_var_c(&sim_mean, sim_arr, size, off_idx), 0.5);

    covar = get_covar_c(act_mean, &sim_mean, act_arr, sim_arr, size, off_idx);
    correl = get_corr_c(act_std_dev, &sim_std_dev, &covar);

    b = sim_mean / *act_mean;
    g = (sim_std_dev / sim_mean) / (*act_std_dev / *act_mean);

    kge = 1 - pow(pow((correl - 1), 2) + \
                  pow((b - 1), 2) + \
                  pow((g - 1), 2), 0.5);
    return kge;
}