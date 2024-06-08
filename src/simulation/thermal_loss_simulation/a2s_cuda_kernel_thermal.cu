#include "../include/a2s_global_function.h"
#include "../include/a2s_device_function.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void thermalKernel(const double* d_i1_data_a2s, const double* d_i2_data_a2s, const double* vc_data_a2s, double *d_thermal_loss) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    d_thermal_loss[tid] = d_i1_data_a2s[tid];

    __syncthreads();
    
} // __global__ void kernel
