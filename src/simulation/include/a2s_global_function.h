#ifndef A2S_GLOBAL_FUNCTION_H
#define A2S_GLOBAL_FUNCTION_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Functions
extern "C" __global__ void avgKernel(const double *S_rated, const int simulation_case, double *d_i1_data_avg, double *d_i2_data_avg, double* d_vc_data_avg);

extern "C" __global__ void a2sKernel(const double *d_i1_data_avg, const double *d_i2_data_avg, const double *d_vc_data_avg, const int simulation_case, double *d_i1_data_a2s, double *d_i2_data_a2s, double *d_vc_data_a2s);

extern "C" __global__ void sumKernel(const double* i1_data, const double* i2_data, const double *vc_data, const int simulation_case, double* i1_mean, double* i2_mean, double* vc_mean);

extern "C" __global__ void adjustKernel(double *d_i1_data_a2s, double *d_i2_data_a2s, double *d_vc_data_a2s, const double *d_i1_mean_avg, const double *d_i2_mean_avg, const double *d_vc_mean_avg, const double *d_i1_mean_a2s, const double *d_i2_mean_a2s, const double *d_vc_mean_a2s);

extern "C" __global__ void thermalKernel(const double* d_i1_data_a2s, const double* d_i2_data_a2s, const double* vc_data_a2s, double *d_thermal_loss);

#endif // A2S_GLOBAL_FUNCTION_H

