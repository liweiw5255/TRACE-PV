#include "../include/a2s_global_function.h"
#include "../include/a2s_device_function.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void a2sKernel(const double *d_i1_data_avg, const double *d_i2_data_avg, const double *d_vc_data_avg, double *d_i1_data_a2s, double *d_i2_data_a2s, double *d_vc_data_a2s) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int case_idx = tid / simulation_size; // Current Case No.
    int time_to_idx = tid % simulation_size; // Current swithicng period for current case
    double initConditionA[DIM], initConditionB[DIM], initConditionC[DIM];
    double duration[duration_sz], outputStatus[duration_sz*DIM];
    double paramsA[2], paramsB[2], paramsC[2];
    double Vr_ll[DIM];
    double current_loop_time, current_time, current_duration, current_state[DIM];
    int current_duration_size, current_loop_size;
    
    
    if(tid < simulation_total_size){
    // if(time_to_idx < simulation_total_size){   
        
        // Calculate current time
        current_time = switching_period * time_to_idx;
        current_loop_time = current_time;
          
        Vr_ll[0] = Vmag * sin(2*M_PI*f*current_time + PhaseShift);
        Vr_ll[1] = Vmag * sin(2*M_PI*f*current_time + 4*M_PI/3 + PhaseShift);
        Vr_ll[2] = Vmag * sin(2*M_PI*f*current_time + 2*M_PI/3 + PhaseShift);
        
        initConditionA[0] = d_i1_data_avg[time_to_idx * duration_size * DIM];
        initConditionA[1] = d_i2_data_avg[time_to_idx * duration_size * DIM];
        initConditionA[2] = d_vc_data_avg[time_to_idx * duration_size * DIM]; 
        
        initConditionB[0] = d_i1_data_avg[time_to_idx * duration_size * DIM + 1];
        initConditionB[1] = d_i2_data_avg[time_to_idx * duration_size * DIM + 1];
        initConditionB[2] = d_vc_data_avg[time_to_idx * duration_size * DIM + 1];
        
        initConditionC[0] = d_i1_data_avg[time_to_idx * duration_size * DIM + 2];
        initConditionC[1] = d_i2_data_avg[time_to_idx * duration_size * DIM + 2];
        initConditionC[2] = d_vc_data_avg[time_to_idx * duration_size * DIM + 2];
        
        getSVM3Phase(Vr_ll, duration, outputStatus);
        
        paramsA[0] = VG_mag * sin(2*M_PI*VG_freq*current_time + VG_phase);
        paramsB[0] = VG_mag * sin(2*M_PI*VG_freq*current_time + 2*M_PI/3 + VG_phase);
        paramsC[0] = VG_mag * sin(2*M_PI*VG_freq*current_time + 4*M_PI/3 + VG_phase);
        
        // Duration size is supposed to be 7
        for(int svm_idx = 0; svm_idx<duration_sz; ++svm_idx){
            current_duration = duration[svm_idx];
            current_duration_size = int(round(current_duration/step_size*100)/100);
            
            for(int i = 0; i < DIM; ++i){
                current_state[i] = outputStatus[svm_idx*DIM+i];
            }

            current_loop_size = int(round(current_loop_time/step_size*100)/100);
            
            if(current_loop_time >= (current_time + switching_period) - 1e-7 && current_loop_time + current_duration >= simulation_time) break;

            if(current_duration > step_size){
                
                paramsA[1] = Vdc * (2.0/3*current_state[0] - 1.0/3*current_state[1] - 1.0/3*current_state[2]);
                paramsB[1] = Vdc * (1.0/3*current_state[0] + 2.0/3*current_state[1] - 1.0/3*current_state[2]);
                paramsC[1] = Vdc * (-1.0/3*current_state[0] - 1.0/3*current_state[1] + 2.0/3*current_state[2]);
                
                rungeKuttaSolve(initConditionA, current_loop_time, current_loop_time + current_duration, step_size, paramsA, d_i1_data_a2s + (case_idx * simulation_size + current_loop_size) * DIM);
                rungeKuttaSolve(initConditionB, current_loop_time, current_loop_time + current_duration, step_size, paramsB, d_i2_data_a2s + (case_idx * simulation_size + current_loop_size) * DIM);
                rungeKuttaSolve(initConditionC, current_loop_time, current_loop_time + current_duration, step_size, paramsC, d_vc_data_a2s + (case_idx * simulation_size + current_loop_size) * DIM);
                
                // Update Initial Cosndition
                initConditionA[0] = d_i1_data_a2s[(current_loop_size) * DIM];
                initConditionA[1] = d_i2_data_a2s[(current_loop_size) * DIM];
                initConditionA[2] = d_vc_data_a2s[(current_loop_size) * DIM];
                
                initConditionB[0] = d_i1_data_a2s[(current_loop_size) * DIM + 1];
                initConditionB[1] = d_i2_data_a2s[(current_loop_size) * DIM + 1];
                initConditionB[2] = d_vc_data_a2s[(current_loop_size) * DIM + 1];

                initConditionC[0] = d_i1_data_a2s[(current_loop_size) * DIM + 2];
                initConditionC[1] = d_i2_data_a2s[(current_loop_size) * DIM + 2];
                initConditionC[2] = d_vc_data_a2s[(current_loop_size) * DIM + 2];
            
            } // if(current_duration > step_size)

            current_loop_time += current_duration;            

        } // for(int svm_idx = 0; svm_idx<7; ++svm_idx)
        
    } // if(time_to_idx < simulation_total_size)  

} // __global__ void kernel
