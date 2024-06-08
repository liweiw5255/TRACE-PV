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

// Macro to determine the maximum of two values
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Kernel launcher function
void launchKernel(double const* S_rated, double* thermal_loss) {
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    
    if (numGPUs < 2) {
        std::cerr << "This example requires at least two GPUs." << std::endl;
        return;
    }

    size_t halfSize = simulation_total_size / 2;
    size_t dataSize = halfSize * duration_size * DIM;
    size_t dataMeanSize = halfSize * DIM;
    size_t requestedSize = 6 * sizeof(double) * simulation_total_size * duration_size * DIM;

    std::cout << "Requested Memory: " << requestedSize / (1024 * 1024 * 1024) << " GB" << std::endl;

    // Arrays to hold device pointers for each GPU
    double *d_i1_data_avg[numGPUs], *d_i2_data_avg[numGPUs], *d_vc_data_avg[numGPUs];
    double *d_i1_data_a2s[numGPUs], *d_i2_data_a2s[numGPUs], *d_vc_data_a2s[numGPUs];
    double *d_i1_mean_a2s[numGPUs], *d_i2_mean_a2s[numGPUs], *d_vc_mean_a2s[numGPUs];
    double *d_i1_mean_avg[numGPUs], *d_i2_mean_avg[numGPUs], *d_vc_mean_avg[numGPUs];
    double *d_S_rated[numGPUs];
    double *d_thermal_loss[numGPUs];

    // Determine the number of blocks needed for different kernels
    int num_blocks_avg = (halfSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_a2s = (halfSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_mean = (halfSize * DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_adjust = (halfSize * DIM + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_thermal = (simulation_case + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << num_blocks_avg << " " << num_blocks_a2s << " " << num_blocks_mean << " " << num_blocks_adjust << " " << num_blocks_thermal << std::endl;

    // Declare events and streams for timing and synchronization
    cudaEvent_t startEvent[numGPUs], stop1Event[numGPUs], stop2Event[numGPUs], stop3Event[numGPUs], stop4Event[numGPUs], stop5Event[numGPUs], stop6Event[numGPUs];
    float avg_time[numGPUs], avg_mean_time[numGPUs], a2s_time[numGPUs], a2s_mean_time[numGPUs], adjust_time[numGPUs], thermal_time[numGPUs];
    cudaStream_t streams[numGPUs];

    for (int i = 0; i < numGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(i));

        // Allocate memory for average model data
        CUDA_CHECK(cudaMalloc(&d_i1_data_avg[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_i2_data_avg[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_vc_data_avg[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_i1_mean_avg[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMalloc(&d_i2_mean_avg[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMalloc(&d_vc_mean_avg[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMalloc(&d_thermal_loss[i], sizeof(double) * simulation_case));
        CUDA_CHECK(cudaMalloc(&d_S_rated[i], sizeof(double) * simulation_case));

        // Allocate memory for A2S model data
        CUDA_CHECK(cudaMalloc(&d_i1_data_a2s[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_i2_data_a2s[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_vc_data_a2s[i], sizeof(double) * dataSize));
        CUDA_CHECK(cudaMalloc(&d_i1_mean_a2s[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMalloc(&d_i2_mean_a2s[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMalloc(&d_vc_mean_a2s[i], sizeof(double) * dataMeanSize));
        CUDA_CHECK(cudaMemcpy(d_S_rated[i], S_rated + i * (simulation_case / numGPUs), sizeof(double) * (simulation_case / numGPUs), cudaMemcpyHostToDevice));
    
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvent[i]));
        CUDA_CHECK(cudaEventCreate(&stop1Event[i]));
        CUDA_CHECK(cudaEventCreate(&stop2Event[i]));
        CUDA_CHECK(cudaEventCreate(&stop3Event[i]));
        CUDA_CHECK(cudaEventCreate(&stop4Event[i]));
        CUDA_CHECK(cudaEventCreate(&stop5Event[i]));
        CUDA_CHECK(cudaEventCreate(&stop6Event[i]));
        
        // Record the start event
        CUDA_CHECK(cudaEventRecord(startEvent[i], streams[i]));

        // Launch the avgKernel
        avgKernel<<<num_blocks_avg, BLOCK_SIZE, 0, streams[i]>>>(d_S_rated[i], d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop1Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop1Event[i]));

        // Launch the sumKernel for average model
        sumKernel<<<num_blocks_mean, BLOCK_SIZE, 0, streams[i]>>>(d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i], d_i1_mean_avg[i], d_i2_mean_avg[i], d_vc_mean_avg[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop2Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop2Event[i]));

        // Launch the a2sKernel
        a2sKernel<<<num_blocks_a2s, BLOCK_SIZE, 0, streams[i]>>>(d_i1_data_avg[i], d_i2_data_avg[i], d_vc_data_avg[i], d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop3Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop3Event[i]));

        // Launch the sumKernel for A2S model
        sumKernel<<<num_blocks_mean, BLOCK_SIZE, 0, streams[i]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_i1_mean_a2s[i], d_i2_mean_a2s[i], d_vc_mean_a2s[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop4Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop4Event[i]));

        // Launch the adjustKernel
        adjustKernel<<<num_blocks_adjust, BLOCK_SIZE, 0, streams[i]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_i1_mean_avg[i], d_i2_mean_avg[i], d_vc_mean_avg[i], d_i1_mean_a2s[i], d_i2_mean_a2s[i], d_vc_mean_a2s[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop5Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop5Event[i]));

        // Launch the thermalKernel
        thermalKernel<<<num_blocks_thermal, BLOCK_SIZE, 0, streams[i]>>>(d_i1_data_a2s[i], d_i2_data_a2s[i], d_vc_data_a2s[i], d_thermal_loss[i]);
        CUDA_CHECK(cudaGetLastError()); // Check for errors after kernel launch
        CUDA_CHECK(cudaEventRecord(stop6Event[i], streams[i]));
        CUDA_CHECK(cudaEventSynchronize(stop6Event[i]));

        // Record elapsed times foreach kernel
        CUDA_CHECK(cudaEventElapsedTime(&avg_time[i], startEvent[i], stop1Event[i]));
        CUDA_CHECK(cudaEventElapsedTime(&avg_mean_time[i], stop1Event[i], stop2Event[i]));
        CUDA_CHECK(cudaEventElapsedTime(&a2s_time[i], stop2Event[i], stop3Event[i]));
        CUDA_CHECK(cudaEventElapsedTime(&a2s_mean_time[i], stop3Event[i], stop4Event[i]));
        CUDA_CHECK(cudaEventElapsedTime(&adjust_time[i], stop4Event[i], stop5Event[i]));
        CUDA_CHECK(cudaEventElapsedTime(&thermal_time[i], stop5Event[i], stop6Event[i]));

        // Output the maximum elapsed times for each kernel
        std::cout << "AVG Model: " << MAX(avg_time[0], avg_time[1]) * 0.001 << " seconds" << std::endl;
        std::cout << "AVG Sum Model: " << MAX(avg_mean_time[0], avg_mean_time[1]) * 0.001 << " seconds" << std::endl;
        std::cout << "A2S Model: " << MAX(a2s_time[0], a2s_time[1]) * 0.001 << " seconds" << std::endl;
        std::cout << "A2S Sum Model: " << MAX(a2s_mean_time[0], a2s_mean_time[1]) * 0.001 << " seconds" << std::endl;
        std::cout << "Adjust Model: " << MAX(adjust_time[0], adjust_time[1]) * 0.001 << " seconds" << std::endl;
        std::cout << "Thermal Model: " << MAX(thermal_time[0], thermal_time[1]) * 0.001 << " seconds" << std::endl;

        // Copy thermal loss results back to host
        CUDA_CHECK(cudaMemcpy(&thermal_loss[i * (simulation_case / numGPUs)], d_thermal_loss[i], sizeof(double) * (simulation_case / numGPUs), cudaMemcpyDeviceToHost));
        for (int j = 0; j < 10; ++j)
            std::cout << thermal_loss[i * (simulation_case / numGPUs) + j] << std::endl;

        // Clean up events and streams
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvent[i]));
        CUDA_CHECK(cudaEventDestroy(stop1Event[i]));
        CUDA_CHECK(cudaEventDestroy(stop2Event[i]));
        CUDA_CHECK(cudaEventDestroy(stop3Event[i]));
        CUDA_CHECK(cudaEventDestroy(stop4Event[i]));
        CUDA_CHECK(cudaEventDestroy(stop5Event[i]));
        CUDA_CHECK(cudaEventDestroy(stop6Event[i]));

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
    }

}