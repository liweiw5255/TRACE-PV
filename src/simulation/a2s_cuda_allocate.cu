#include "include/a2s_header.h"
#include "include/a2s_parameters.h"
#include <iostream>
#include <chrono>

// Macro to check CUDA calls and report errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error calling \""#call"\", code is " << err << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro to check CUBLAS calls and report errors
#define CUBLAS_CHECK(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error calling \""#call"\"" << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel launcher function
void launchKernel(double const* S_rated, double* thermal_loss) {
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));

    size_t index = 0;
    size_t singleCaseSize = 6 * simulation_size * duration_size * DIM + 6 * simulation_size * DIM + 2 * simulation_case * DIM;
    size_t freeMem[numGPUs], totalMem[numGPUs], requestedSize[numGPUs], local_simulation_case_num[numGPUs], dataSize[numGPUs], dataMeanSize[numGPUs];
    size_t num_blocks_avg[numGPUs], num_blocks_a2s[numGPUs], num_blocks_mean[numGPUs], num_blocks_adjust[numGPUs], num_blocks_thermal[numGPUs];

    // Arrays to hold device pointers for each GPU
    double *d_i1_data_avg[numGPUs], *d_i2_data_avg[numGPUs], *d_vc_data_avg[numGPUs];
    double *d_i1_data_a2s[numGPUs], *d_i2_data_a2s[numGPUs], *d_vc_data_a2s[numGPUs];
    double *d_i1_mean_a2s[numGPUs], *d_i2_mean_a2s[numGPUs], *d_vc_mean_a2s[numGPUs];
    double *d_i1_mean_avg[numGPUs], *d_i2_mean_avg[numGPUs], *d_vc_mean_avg[numGPUs];
    double *d_S_rated[numGPUs];
    double *d_thermal_loss[numGPUs];

    // Declare events and streams for timing and synchronization
    cudaEvent_t startEvent[numGPUs], stopEvent[numGPUs][6];
    float times[numGPUs][6];
    cudaStream_t streams[numGPUs][2];

    while (index < simulation_case) {
        for (int i = 0; i < numGPUs; ++i) {
            CUDA_CHECK(cudaSetDevice(i));

            CUDA_CHECK(cudaMemGetInfo(&freeMem[i], &totalMem[i]));

            local_simulation_case_num[i] = freeMem[i] / singleCaseSize / sizeof(double);        
            local_simulation_case_num[i] = (i == numGPUs - 1) ? std::min(local_simulation_case_num[i], simulation_case - index) : local_simulation_case_num[i];

            dataSize[i] = local_simulation_case_num[i] * simulation_size * duration_size * DIM;
            dataMeanSize[i] = local_simulation_case_num[i] * simulation_size * DIM;
            requestedSize[i] = local_simulation_case_num[i] * singleCaseSize * sizeof(double);

            std::cout << "GPU " << i << " Requested Memory: " << double(requestedSize[i]) / (1024 * 1024 * 1024) << " GB, Free Memory: " << double(freeMem[i]) / (1024 * 1024 * 1024) << " GB " << index << std::endl;
            
            // Determine the number of blocks needed for different kernels
            num_blocks_avg[i] = (local_simulation_case_num[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks_a2s[i] = (dataSize[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks_mean[i] = (dataMeanSize[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks_adjust[i] = (dataSize[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks_thermal[i] = (local_simulation_case_num[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Allocate memory for average model data
            CUDA_CHECK(cudaMalloc(&d_i1_data_avg[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i2_data_avg[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_vc_data_avg[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i1_mean_avg[i], sizeof(double) * dataMeanSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i2_mean_avg[i], sizeof(double) * dataMeanSize[i]));
            CUDA_CHECK(cudaMalloc(&d_vc_mean_avg[i], sizeof(double) * dataMeanSize[i]));
            CUDA_CHECK(cudaMalloc(&d_thermal_loss[i], sizeof(double) * local_simulation_case_num[i]));
            CUDA_CHECK(cudaMalloc(&d_S_rated[i], sizeof(double) * local_simulation_case_num[i]));
       
            // Allocate memory for A2S model data
            CUDA_CHECK(cudaMalloc(&d_i1_data_a2s[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i2_data_a2s[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_vc_data_a2s[i], sizeof(double) * dataSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i1_mean_a2s[i], sizeof(double) * dataMeanSize[i]));
            CUDA_CHECK(cudaMalloc(&d_i2_mean_a2s[i], sizeof(double) * dataMeanSize[i]));
            CUDA_CHECK(cudaMalloc(&d_vc_mean_a2s[i], sizeof(double) * dataMeanSize[i]));
            
            CUDA_CHECK(cudaStreamCreate(&streams[i][0]));
            CUDA_CHECK(cudaStreamCreate(&streams[i][1]));
            CUDA_CHECK(cudaEventCreate(&startEvent[i]));

             std::cout << "Copying No." << index  << " cases to GPU " << i << std::endl;
            CUDA_CHECK(cudaMemcpyAsync(d_S_rated[i], S_rated + index, sizeof(double) * local_simulation_case_num[i], cudaMemcpyHostToDevice, streams[i][0]));
            for (int j = 0; j < 6; ++j) {
                CUDA_CHECK(cudaEventCreate(&stopEvent[i][j]));
            }

            // Record the start event
            CUDA_CHECK(cudaEventRecord(startEvent[i], streams[i][0]));

            // Launch the avgKernel
            avgKernel<<<num_blocks_avg[i], BLOCK_SIZE, 0, streams[i][0]>>>(d_S_rated[i], d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][0], streams[i][0]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i][0])); // Ensure the kernel has finished

            // Launch the sumKernel for average model
            sumKernel<<<num_blocks_mean[i], BLOCK_SIZE, 0, streams[i][1]>>>(d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i], d_i1_mean_avg[i], d_i2_mean_avg[i], d_vc_mean_avg[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][1], streams[i][1]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i][1])); // Ensure the kernel has finished

            // Launch the a2sKernel
            a2sKernel<<<num_blocks_a2s[i], BLOCK_SIZE, 0, streams[i][0]>>>(d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i], d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][2], streams[i][0]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i][0])); // Ensure the kernel has finished

            // Launch the sumKernel for A2S model
            sumKernel<<<num_blocks_mean[i], BLOCK_SIZE, 0, streams[i][1]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_i1_mean_a2s[i], d_i2_mean_a2s[i], d_vc_mean_a2s[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][3], streams[i][1]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i][1])); // Ensure the kernel has finished

            // Launch the adjustKernel
            adjustKernel<<<num_blocks_adjust[i], BLOCK_SIZE, 0, streams[i][0]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_i1_mean_avg[i], d_i2_mean_avg[i], d_vc_mean_avg[i], d_i1_mean_a2s[i], d_i2_mean_a2s[i], d_vc_mean_a2s[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][4], streams[i][0]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i])); // Ensure the kernel has finished

            // Launch the thermalKernel
            thermalKernel<<<num_blocks_thermal[i], BLOCK_SIZE, 0, streams[i][1]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_thermal_loss[i]);
            CUDA_CHECK(cudaEventRecord(stopEvent[i][5], streams[i][1]));
            // CUDA_CHECK(cudaStreamSynchronize(streams[i])); // Ensure the kernel has finished

            // Copy thermal loss results back to host
            CUDA_CHECK(cudaMemcpyAsync(&thermal_loss[index], d_thermal_loss[i], sizeof(double) * local_simulation_case_num[i], cudaMemcpyDeviceToHost, streams[i][0]));

            // Record elapsed times for each kernel
            for (int j = 0; j < 6; ++j) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i][j % 2])); // Synchronize the appropriate stream
                CUDA_CHECK(cudaEventElapsedTime(&times[i][j], j == 0 ? startEvent[i] : stopEvent[i][j - 1], stopEvent[i][j]));
            }

            // Output the elapsed times for each kernel
            std::cout << "GPU " << i << " AVG Model: " << times[i][0] * 0.001 << " seconds" << std::endl;
            std::cout << "GPU " << i << " AVG Sum Model: " << times[i][1] * 0.001 << " seconds" << std::endl;
            std::cout << "GPU " << i << " A2S Model: " << times[i][2] * 0.001 << " seconds" << std::endl;
            std::cout << "GPU " << i << " A2S Sum Model: " << times[i][3] * 0.001 << " seconds" << std::endl;
            std::cout << "GPU " << i << " Adjust Model: " << times[i][4] * 0.001 << " seconds" << std::endl;
            std::cout << "GPU " << i << " Thermal Model: " << times[i][5] * 0.001 << " seconds" << std::endl;

                        
            for(int k=0; k<10; ++k){
                std::cout << thermal_loss[k] << std::endl;
            }

            // Free device memory
            CUDA_CHECK(cudaFree(d_i1_data_a2s[i]));
            CUDA_CHECK(cudaFree(d_i2_data_a2s[i]));
            CUDA_CHECK(cudaFree(d_vc_data_a2s[i]));
            CUDA_CHECK(cudaFree(d_i1_data_avg[i]));
            CUDA_CHECK(cudaFree(d_i2_data_avg[i]));
            CUDA_CHECK(cudaFree(d_vc_data_avg[i]));
            CUDA_CHECK(cudaFree(d_i1_mean_avg[i]));
            CUDA_CHECK(cudaFree(d_i2_mean_avg[i]));
            CUDA_CHECK(cudaFree(d_vc_mean_avg[i]));
            CUDA_CHECK(cudaFree(d_i1_mean_a2s[i]));
            CUDA_CHECK(cudaFree(d_i2_mean_a2s[i]));
            CUDA_CHECK(cudaFree(d_vc_mean_a2s[i]));

            index += local_simulation_case_num[i];
        } // Iterate GPU

    } // While (Index < Simulation Case)

    for (int i = 0; i < numGPUs; ++i) {
        // Clean up events and streams
        CUDA_CHECK(cudaStreamDestroy(streams[i][0]));
        CUDA_CHECK(cudaStreamDestroy(streams[i][1]));
        CUDA_CHECK(cudaEventDestroy(startEvent[i]));
        for (int j = 0; j < 6; ++j) {
            CUDA_CHECK(cudaEventDestroy(stopEvent[i][j]));
        }
    }

} // Function End