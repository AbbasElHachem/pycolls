#include <math.h>
#include <stdio.h>

double min(double a, double b){
    if (a <= b) {
    return a;
    }
    else {
    return b;
    }
}

double max(double a, double b){
    if (a >= b) {
    return a;
    }
    else {
    return b;
    }
}

//TODO: add random function later

double loop_HBV_c(const unsigned long *n_recs,
                  const double *conv_ratio, const double *prms_arr,
                  const double *ini_arr, const double *temp_arr,
                  const double *prec_arr, const double *pet_arr,
                  double *q_sim_arr, double *out_arr,
                  const unsigned long *n_ocols) {
    unsigned long i, j = 0;
    double temp, snow, prec, liqu, sm;

    double tt = prms_arr[0];
    double c_melt = prms_arr[1];
    double fc = prms_arr[2];
    double beta = prms_arr[3];
    double pwp = prms_arr[4];
    double ur_thresh = prms_arr[5];
    double k_uu = prms_arr[6];
    double k_ul = prms_arr[7];
    double k_d = prms_arr[8];
    double k_ll = prms_arr[9];

    out_arr[0] = ini_arr[0];
    out_arr[2] = ini_arr[1];
    out_arr[7] = ini_arr[2];
    out_arr[11] = ini_arr[3];

for (i=*n_ocols; i<((*n_recs * *n_ocols) + *n_ocols); i=i+*n_ocols){
    temp = temp_arr[j];
    snow = out_arr[i - *n_ocols];
    prec = prec_arr[j];

    if (temp < tt) {
        out_arr[i] = snow + prec;
        out_arr[1+i] = 0.0;
    }
    else {
        out_arr[i] = max(0.0, snow - (c_melt * (temp - tt)));
        out_arr[1+i] = prec + min(snow, (c_melt * (temp - tt)));
    }

    liqu = out_arr[1+i];
    sm = out_arr[2+(i-*n_ocols)];

    if (sm > pwp){
        out_arr[4+i] = pet_arr[j];
    }
    else {
        out_arr[4+i] = (sm / fc) * pet_arr[j];
    }

    if ((sm < 0) || (fc <= 0)) {
        return -1e5;
    }

    out_arr[2+i] = sm - out_arr[4+i] + (liqu * (1 - pow((sm / fc), beta)));

    out_arr[3+i] = liqu * pow((sm / fc), beta);

    out_arr[8+i] = max(0.0, k_uu * (out_arr[7+(i-*n_ocols)] - ur_thresh));

    out_arr[10+i] = (out_arr[7+(i-*n_ocols)] - out_arr[8+i]) * k_d;

    out_arr[9+i] = max(0.0, (out_arr[7+(i-*n_ocols)] - out_arr[10+i]) * k_ul);

    out_arr[7+i] = max(0.0, (out_arr[7+(i-*n_ocols)] + \
                             out_arr[3+i] - \
                             out_arr[8+i] - \
                             out_arr[9+i] - \
                             out_arr[10+i]));

    out_arr[12+i] = out_arr[11+(i-*n_ocols)] * k_ll;

    out_arr[11+i] = out_arr[10+i] + out_arr[11+(i-*n_ocols)] - out_arr[12+i];

    out_arr[5+i] = \
    out_arr[8+i] + \
    out_arr[9+i] + \
    out_arr[12+i];

    out_arr[6+i] = *conv_ratio * \
    out_arr[5+i];

    q_sim_arr[j] = \
    out_arr[6+i];

    j = j + 1;

}

return 0.0;
}
