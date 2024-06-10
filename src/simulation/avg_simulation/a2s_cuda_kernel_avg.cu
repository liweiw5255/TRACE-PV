#include "../include/a2s_global_function.h"
#include "../include/a2s_device_function.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void avgKernel(const double* d_S_rated, const int simulation_case, double* d_i1_data_avg, double* d_i2_data_avg, double* d_vc_data_avg) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < simulation_case) {
        cuFloatComplex Z_L1, Z_C, Z_L2, I_L2, V_C, I_C, I_L1, V_L1, V_A; 
        double M_IL1, M_IL2, Phase_L1, Phase_L2, M_VC, Phase_V;

        Z_L1 = make_cuFloatComplex(RL1, omega * L1);
        Z_C = make_cuFloatComplex(Rc, -1.0f / (omega * C));
        Z_L2 = make_cuFloatComplex(RL2, omega * L2);
        
        I_L2 = make_cuFloatComplex(d_S_rated[idx] / (3 * V_g * n2 / n1), 0);
        V_C = cuCaddf(make_cuFloatComplex(V_g * n2 / n1, 0), cuCmulf(I_L2, Z_L2));
        I_C = cuCdivf(V_C, Z_C);
        I_L1 = cuCaddf(I_C, I_L2);
        V_L1 = cuCmulf(I_L1, Z_L1);
        V_A = cuCaddf(V_C, V_L1);
        
        M_IL1 = cuCabsf(I_L1);
        Phase_L1 = atan2f(cuCimagf(I_L1), cuCrealf(I_L1));
        M_IL2 = cuCabsf(I_L2);
        Phase_L2 = atan2f(cuCimagf(I_L2), cuCrealf(I_L2));
        M_VC = cuCabsf(V_C);
        Phase_V = atan2f(cuCimagf(V_C), cuCrealf(V_C));

        for (int local_idx = 0; local_idx < duration_size; ++local_idx) {
            int base_idx1 = idx * DIM * duration_size + local_idx;
            int base_idx2 = (idx * DIM + 1) * duration_size + local_idx;
            int base_idx3 = (idx * DIM + 2) * duration_size + local_idx;

            d_i1_data_avg[base_idx1] = M_IL1 * cosf(omega * local_idx * step_size + Phase_L1);
            d_i1_data_avg[base_idx2] = M_IL1 * cosf(omega * local_idx * step_size + Phase_L1 - 2.0 * M_PI / 3);
            d_i1_data_avg[base_idx3] = M_IL1 * cosf(omega * local_idx * step_size + Phase_L1 + 2.0 * M_PI / 3);

            d_i2_data_avg[base_idx1] = M_IL2 * cosf(omega * local_idx * step_size + Phase_L2);
            d_i2_data_avg[base_idx2] = M_IL2 * cosf(omega * local_idx * step_size + Phase_L2 - 2.0 * M_PI / 3);
            d_i2_data_avg[base_idx3] = M_IL2 * cosf(omega * local_idx * step_size + Phase_L2 + 2.0 * M_PI / 3);

            d_vc_data_avg[base_idx1] = M_VC * cosf(omega * local_idx * step_size + Phase_V);
            d_vc_data_avg[base_idx2] = M_VC * cosf(omega * local_idx * step_size + Phase_V - 2.0 * M_PI / 3);
            d_vc_data_avg[base_idx3] = M_VC * cosf(omega * local_idx * step_size + Phase_V + 2.0 * M_PI / 3);
        }
    }
   
    __syncthreads();
    
} // __global__ void kernel
