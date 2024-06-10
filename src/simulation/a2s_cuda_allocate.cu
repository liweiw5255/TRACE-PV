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
void launchKernel(const double* S_rated, const int local_simulation_case, const int offset, double* thermal_loss, MPI_Comm comm) {

    int i, size;
    MPI_Comm_rank(comm, &i);
    MPI_Comm_size(comm, &size);

    // CUDA_CHECK(cudaGetDeviceCount(&numGPUs));

    int index = 0, local_simulation_case_num;
    size_t singleCaseSize = 6 * simulation_size * duration_size * DIM + 6 * simulation_size * DIM + 2 * batchSize * DIM;
    size_t freeMem, totalMem, requestedSize, dataSize, dataMeanSize;
    size_t num_blocks_avg, num_blocks_a2s, num_blocks_mean, num_blocks_adjust, num_blocks_thermal;

    // Arrays to hold device pointers for each GPU
    double *d_i1_data_avg, *d_i2_data_avg, *d_vc_data_avg;
    double *d_i1_data_a2s, *d_i2_data_a2s, *d_vc_data_a2s;
    double *d_i1_mean_a2s, *d_i2_mean_a2s, *d_vc_mean_a2s;
    double *d_i1_mean_avg, *d_i2_mean_avg, *d_vc_mean_avg;
    double *d_S_rated;
    double *d_thermal_loss;

    // Declare events and streams for timing and synchronization
    cudaEvent_t startEvent, stopEvent[6];
    double times[6];
    cudaStream_t streams[2];

    // Set No. i GPU as the current device
    CUDA_CHECK(cudaSetDevice(i));

    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    local_simulation_case_num = freeMem / singleCaseSize / sizeof(double);        
    local_simulation_case_num = std::min(local_simulation_case_num, local_simulation_case - index);

    dataSize = local_simulation_case_num * simulation_size * duration_size * DIM;
    dataMeanSize = local_simulation_case_num * simulation_size * DIM;
    requestedSize = local_simulation_case_num * singleCaseSize * sizeof(double);

    // std::cout << "GPU " << i << " Requested Memory: " << double(requestedSize) / (1024 * 1024 * 1024) << " GB, Free Memory: " << double(freeMem) / (1024 * 1024 * 1024) << " GB" << std::endl;
    
    // Determine the number of blocks needed for different kernels
    num_blocks_avg = (local_simulation_case_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_a2s = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_mean = (dataMeanSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_adjust = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_thermal = (local_simulation_case_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate memory for average model data
    CUDA_CHECK(cudaMalloc(&d_i1_data_avg, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_i2_data_avg, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_vc_data_avg, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_i1_mean_avg, sizeof(double) * dataMeanSize));
    CUDA_CHECK(cudaMalloc(&d_i2_mean_avg, sizeof(double) * dataMeanSize));
    CUDA_CHECK(cudaMalloc(&d_vc_mean_avg, sizeof(double) * dataMeanSize));
    CUDA_CHECK(cudaMalloc(&d_thermal_loss, sizeof(double) * local_simulation_case_num));
    CUDA_CHECK(cudaMalloc(&d_S_rated, sizeof(double) * local_simulation_case_num));

    // Allocate memory for A2S model data
    CUDA_CHECK(cudaMalloc(&d_i1_data_a2s, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_i2_data_a2s, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_vc_data_a2s, sizeof(double) * dataSize));
    CUDA_CHECK(cudaMalloc(&d_i1_mean_a2s, sizeof(double) * dataMeanSize));
    CUDA_CHECK(cudaMalloc(&d_i2_mean_a2s, sizeof(double) * dataMeanSize));
    CUDA_CHECK(cudaMalloc(&d_vc_mean_a2s, sizeof(double) * dataMeanSize));
    
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    CUDA_CHECK(cudaEventCreate(&startEvent));

    while(index < batchSize){
 
        // std::cout << "Copying No." << offset + index  << " cases to GPU " << i << std::endl;
        CUDA_CHECK(cudaMemcpyAsync(d_S_rated, S_rated, sizeof(double) * local_simulation_case_num, cudaMemcpyHostToDevice, streams[0]));
        for (int j = 0; j < 6; ++j) {
            CUDA_CHECK(cudaEventCreate(&stopEvent[j]));
        }

        // Record the start event
        CUDA_CHECK(cudaEventRecord(startEvent, streams[0]));

        // Launch the avgKernel
        avgKernel<<<num_blocks_avg, BLOCK_SIZE, 0, streams[0]>>>(d_S_rated, local_simulation_case_num, d_i1_data_avg, d_i2_data_avg, d_vc_data_avg);
        CUDA_CHECK(cudaEventRecord(stopEvent[0], streams[0]));

        
        // Launch the sumKernel for average model
        sumKernel<<<num_blocks_mean, BLOCK_SIZE, 0, streams[0]>>>(d_i1_data_avg, d_i2_data_avg, d_vc_data_avg, local_simulation_case, d_i1_mean_avg, d_i2_mean_avg, d_vc_mean_avg);
        CUDA_CHECK(cudaEventRecord(stopEvent[1], streams[0]));

        // Launch the a2sKernel
        a2sKernel<<<num_blocks_a2s, BLOCK_SIZE, 0, streams[0]>>>(d_i1_data_avg, d_i2_data_avg, d_vc_data_avg, local_simulation_case_num, d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s);
        CUDA_CHECK(cudaEventRecord(stopEvent[2], streams[0]));

        // Launch the sumKernel for A2S model
        sumKernel<<<num_blocks_mean, BLOCK_SIZE, 0, streams[0]>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, local_simulation_case, d_i1_mean_a2s, d_i2_mean_a2s, d_vc_mean_a2s);
        CUDA_CHECK(cudaEventRecord(stopEvent[3], streams[0]));

        // Launch the adjustKernel
        adjustKernel<<<num_blocks_adjust, BLOCK_SIZE, 0, streams[0]>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, d_i1_mean_avg, d_i2_mean_avg, d_vc_mean_avg, d_i1_mean_a2s, d_i2_mean_a2s, d_vc_mean_a2s);
        CUDA_CHECK(cudaEventRecord(stopEvent[4], streams[0]));

        // Launch the thermalKernel
        thermalKernel<<<num_blocks_thermal, BLOCK_SIZE, 0, streams[0]>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, d_thermal_loss);
        CUDA_CHECK(cudaEventRecord(stopEvent[5], streams[0]));

        // Copy thermal loss results back to host
        CUDA_CHECK(cudaMemcpyAsync(thermal_loss, d_thermal_loss, sizeof(double) * local_simulation_case_num, cudaMemcpyDeviceToHost, streams[0]));
        
        // Record elapsed times for each kernel
        /*
        for (int j = 0; j < 6; ++j) {
            CUDA_CHECK(cudaStreamSynchronize(streams[j % 2])); // Synchronize the appropriate stream
            CUDA_CHECK(cudaEventElapsedTime(&times[j], j == 0 ? startEvent : stopEvent[j - 1], stopEvent[j]));
        }

        // Output the elapsed times for each kernel
        std::cout << "GPU " << i << " AVG Model: " << times[0] * 0.001 << " seconds" << std::endl;
        std::cout << "GPU " << i << " AVG Sum Model: " << times[1] * 0.001 << " seconds" << std::endl;
        std::cout << "GPU " << i << " A2S Model: " << times[2] * 0.001 << " seconds" << std::endl;
        std::cout << "GPU " << i << " A2S Sum Model: " << times[3] * 0.001 << " seconds" << std::endl;
        std::cout << "GPU " << i << " Adjust Model: " << times[4] * 0.001 << " seconds" << std::endl;
        std::cout << "GPU " << i << " Thermal Model: " << times[5] * 0.001 << " seconds" << std::endl;
        
        for(int k=0; k<10; ++k){
            std::cout << thermal_loss[k] << std::endl;
        }
        */

        index += local_simulation_case_num;

    } // While (index < batchSize)

    // Free device memory
    CUDA_CHECK(cudaFree(d_i1_data_avg));
    CUDA_CHECK(cudaFree(d_i2_data_avg));
    CUDA_CHECK(cudaFree(d_vc_data_avg));

    CUDA_CHECK(cudaFree(d_i1_mean_avg));
    CUDA_CHECK(cudaFree(d_i2_mean_avg));
    CUDA_CHECK(cudaFree(d_vc_mean_avg));

    CUDA_CHECK(cudaFree(d_i1_data_a2s));
    CUDA_CHECK(cudaFree(d_i2_data_a2s));
    CUDA_CHECK(cudaFree(d_vc_data_a2s));

    CUDA_CHECK(cudaFree(d_i1_mean_a2s));
    CUDA_CHECK(cudaFree(d_i2_mean_a2s));
    CUDA_CHECK(cudaFree(d_vc_mean_a2s));

    // Clean up events and streams
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    for (int j = 0; j < 6; ++j) {
        CUDA_CHECK(cudaEventDestroy(stopEvent[j]));
    }

    MPI_Barrier(comm);

} // Function End