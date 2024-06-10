#include "../include/a2s_global_function.h"
#include "../include/a2s_device_function.h"
#include <cuda_runtime.h>
              
__global__ void sumKernel(const double* i1_data, const double* i2_data, const double *vc_data, const int simulation_case, double* i1_mean, double* i2_mean, double* vc_mean) {

    __shared__ double sharedData_i1[BLOCK_SIZE]; 
    __shared__ double sharedData_i2[BLOCK_SIZE]; 
    __shared__ double sharedData_vc[BLOCK_SIZE]; 
    
    int tid = threadIdx.x;
    int bid = blockIdx.x; // blockIdx.y * gridDim.x + blockIdx.x;

    int num_block = floor(double(duration_size / BLOCK_SIZE));
    int total_num_block = num_block * DIM; // Number of block for each case.
    int num_sw = bid / total_num_block; 
    int sw_id = bid % total_num_block;
    int dim_sw_id = sw_id % DIM;
    int local_sw_id = sw_id / DIM;

    int global_id = (num_sw * total_num_block * BLOCK_SIZE) + (local_sw_id * DIM + dim_sw_id) * BLOCK_SIZE + DIM * tid + dim_sw_id;
    
    if(bid < simulation_case * simulation_size * num_block){
        // sharedData_i1[tid] = (local_sw_id != num_block - 1 && tid < num_block * BLOCK_SIZE - local_sw_id * BLOCK_SIZE)? i1_data[global_id] : 0.0;
        sharedData_i1[tid] = (local_sw_id == num_block - 1 && tid < duration_size - (num_block - 1) * BLOCK_SIZE)? i1_data[global_id] : 0.0; // ((local_sw_id + 1) * BLOCK_SIZE < duration_size)? i1_data[global_id] : 0.0;
        sharedData_i2[tid] = (local_sw_id == num_block - 1 && tid < duration_size - (num_block - 1) * BLOCK_SIZE)? i2_data[global_id] : 0.0; // ((local_sw_id + 1) * BLOCK_SIZE < duration_size)? i2_data[global_id] : 0.0;
        sharedData_vc[tid] = (local_sw_id == num_block - 1 && tid < duration_size - (num_block - 1) * BLOCK_SIZE)? vc_data[global_id] : 0.0; // ((local_sw_id + 1) * BLOCK_SIZE < duration_size)? vc_data[global_id] : 0.0;
            
        __syncthreads();

        // Intra-block reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sharedData_i1[tid] += sharedData_i1[tid + s];
                sharedData_i2[tid] += sharedData_i2[tid + s];
                sharedData_vc[tid] += sharedData_vc[tid + s];
            }
            __syncthreads();
        }

        // Write the result of this block to the output array
        if (tid == 0) {
            i1_mean[bid] = sharedData_i1[0];
            i2_mean[bid] = sharedData_i2[0];
            vc_mean[bid] = sharedData_vc[0];
        }
        __syncthreads();
    }
}



__global__ void adjustKernel(double *d_i1_data_a2s, double *d_i2_data_a2s, double *d_vc_data_a2s, const double *d_i1_mean_avg, const double *d_i2_mean_avg, const double *d_vc_mean_avg, const double *d_i1_mean_a2s, const double *d_i2_mean_a2s, const double *d_vc_mean_a2s){

    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if(index < simulation_size * DIM){
        for(int j=0; j<duration_size; ++j){
            d_i1_data_a2s[index * duration_size + j] += (d_i1_mean_avg[int(index/duration_size)] - d_i1_mean_a2s[int(index/duration_size)]) / duration_size;
            d_i2_data_a2s[index * duration_size + j] += (d_i2_mean_avg[int(index/duration_size)] - d_i2_mean_a2s[int(index/duration_size)]) / duration_size;
            d_vc_data_a2s[index * duration_size + j] += (d_vc_mean_avg[int(index/duration_size)] - d_vc_mean_a2s[int(index/duration_size)]) / duration_size;
        }
    }
}