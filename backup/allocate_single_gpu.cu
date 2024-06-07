#include "include/a2s_header.h"
#include "include/a2s_parameters.h"
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA error calling \""#call"\", code is " << err << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error calling \""#call"\"" << std::endl; \
        exit(EXIT_FAILURE); \
    }

void launchKernel(double const* S_rated, double* thermal_loss) {
    
    double *d_i1_data_avg, *d_i2_data_avg, *d_vc_data_avg;
    double *d_i1_data_a2s, *d_i2_data_a2s, *d_vc_data_a2s; 
    double *d_i1_mean_a2s, *d_i2_mean_a2s, *d_vc_mean_a2s;
    double *d_i1_mean_avg, *d_i2_mean_avg, *d_vc_mean_avg;
    double *d_S_rated;
    double *d_thermal_loss;

    double *data = new double[simulation_total_size * duration_size * DIM];

    int num_blocks_avg, num_blocks_a2s, num_blocks_mean, num_blocks_adjust, num_blocks_thermal, numGPUs = 0;

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    size_t freeMem = 0, currFreeMem, totalMem = 0, currTotalMem;
    
    for(int i=0; i<numGPUs; ++i){
    	CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, i)); // Assuming you're using the first GPU

    	// Query GPU memory
    	CUDA_CHECK(cudaMemGetInfo(&currFreeMem, &currTotalMem));
    	// std::cout << "numGPUs: " << numGPUs << " maxThreadsDim " << deviceProp.maxThreadsDim[0] << " " << deviceProp.maxThreadsDim[1] << " " << deviceProp.maxThreadsDim[2] << std::endl;     
    	
	freeMem += currFreeMem;
	totalMem += currTotalMem;
    }


   size_t requestedSize = 6 * sizeof(double) * simulation_total_size * duration_size * DIM;

    std::cout << " Requested Memory: " << requestedSize / (1024 * 1024 * 1024) << " GB, Free memory: " << freeMem / (1024 * 1024 * 1024) << " GB, Total memory: " << totalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
    

    if (freeMem < requestedSize) {
        std::cerr << "Not enough memory available to allocate the array." << std::endl;
        return;
    }

    // Allocate memory for average model data
    CUDA_CHECK(cudaMalloc(&d_i1_data_avg, sizeof(double) * simulation_total_size * duration_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_i2_data_avg, sizeof(double) * simulation_total_size * duration_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_vc_data_avg, sizeof(double) * simulation_total_size * duration_size * DIM));
    
    CUDA_CHECK(cudaMalloc(&d_i1_mean_avg, sizeof(double) * simulation_total_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_i2_mean_avg, sizeof(double) * simulation_total_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_vc_mean_avg, sizeof(double) * simulation_total_size * DIM));
    
    CUDA_CHECK(cudaMalloc(&d_thermal_loss, sizeof(double) * simulation_case));
    CUDA_CHECK(cudaMalloc(&d_S_rated, sizeof(double) * simulation_case));

    // Allocate memory for a2s model data
    CUDA_CHECK(cudaMalloc(&d_i1_data_a2s, sizeof(double) * simulation_total_size * duration_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_i2_data_a2s, sizeof(double) * simulation_total_size * duration_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_vc_data_a2s, sizeof(double) * simulation_total_size * duration_size * DIM));
    
    CUDA_CHECK(cudaMalloc(&d_i1_mean_a2s, sizeof(double) * simulation_total_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_i2_mean_a2s, sizeof(double) * simulation_total_size * DIM));
    CUDA_CHECK(cudaMalloc(&d_vc_mean_a2s, sizeof(double) * simulation_total_size * DIM));
        
    CUDA_CHECK(cudaMemcpy(d_S_rated, S_rated, sizeof(double) * simulation_case, cudaMemcpyHostToDevice));

    // Allocate resources for device
    num_blocks_avg = (simulation_total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks_a2s = (simulation_total_size + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    num_blocks_mean = simulation_total_size * DIM * floor(duration_size/BLOCK_SIZE); 
    num_blocks_adjust = (simulation_total_size * DIM + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    num_blocks_thermal = (simulation_case + BLOCK_SIZE - 1) / BLOCK_SIZE; 

    std::cout << num_blocks_avg << " " << num_blocks_a2s << " " << num_blocks_mean << " " << num_blocks_adjust << " " << num_blocks_thermal << std::endl;

    dim3 avgDim((num_blocks_avg + 1023) / 1024, 1024);
    dim3 a2sDim((num_blocks_a2s + 1023) / 1024, 1024);
    dim3 meanDim((num_blocks_mean + 1023) / 1024, 1024);
    dim3 adjustDim((num_blocks_adjust + 1023) / 1024, 1024);

    // Declare events
    cudaEvent_t start, stop1, stop2, stop3, stop4, stop5, stop6;
    float avg_time, avg_mean_time, a2s_time, a2s_mean_time, adjust_time, thermal_time;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&stop2));
    CUDA_CHECK(cudaEventCreate(&stop3));
    CUDA_CHECK(cudaEventCreate(&stop4));
    CUDA_CHECK(cudaEventCreate(&stop5));
    CUDA_CHECK(cudaEventCreate(&stop6));

    // Create a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, stream));
    avgKernel<<<num_blocks_avg, BLOCK_SIZE, 0, stream>>>(d_S_rated, d_i1_data_avg, d_i2_data_avg, d_vc_data_avg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop1, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop1));

    // CUDA_CHECK(cudaMemcpy(data, d_i1_data_avg, sizeof(double) * simulation_total_size * duration_size * DIM, cudaMemcpyDeviceToHost));
    // for(int i=0; i<duration_size; ++i)
    //     std::cout << data[i] << std::endl;
  
    sumKernel<<<num_blocks_mean, BLOCK_SIZE, 0, stream>>>(d_i1_data_avg, d_i2_data_avg, d_vc_data_avg, d_i1_mean_avg, d_i2_mean_avg, d_vc_mean_avg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop2, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop2));

    // CUDA_CHECK(cudaMemcpy(data, d_i1_mean_avg, sizeof(double) * simulation_total_size * DIM, cudaMemcpyDeviceToHost));
    // for(int i=0; i<10; ++i)
    //    std::cout << data[i] << std::endl;

    a2sKernel<<<num_blocks_a2s, BLOCK_SIZE, 0, stream>>>(d_i1_data_avg, d_i2_data_avg, d_vc_data_avg, d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s);
    CUDA_CHECK(cudaEventRecord(stop3, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop3));

    // CUDA_CHECK(cudaMemcpy(data, d_i1_data_a2s, sizeof(double) * simulation_total_size * duration_size * DIM, cudaMemcpyDeviceToHost));
    // for(int i=0; i<duration_size; ++i)
    //     std::cout << data[i] << std::endl;

    // Calculate mean value for a2s data
    sumKernel<<<num_blocks_mean, BLOCK_SIZE>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, d_i1_mean_a2s, d_i2_mean_a2s, d_vc_mean_a2s);
    CUDA_CHECK(cudaEventRecord(stop4, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop4));

    // Adjust a2s data based on mean value
    adjustKernel<<<num_blocks_adjust, BLOCK_SIZE>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, d_i1_mean_avg, d_i2_mean_avg, d_vc_mean_avg, d_i1_mean_a2s, d_i2_mean_a2s, d_vc_mean_a2s);
    CUDA_CHECK(cudaEventRecord(stop5, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop5));

    //CUDA_CHECK(cudaMemcpy(data, d_i1_data_a2s, sizeof(double) * simulation_total_size * duration_size * DIM, cudaMemcpyDeviceToHost));
    //for(int i=0; i<duration_size; ++i)
    //    std::cout << data[i] << std::endl;

    // Calculate thermal loss based on switching simulation 
    thermalKernel<<<num_blocks_thermal, BLOCK_SIZE>>>(d_i1_data_a2s, d_i2_data_a2s, d_vc_data_a2s, d_thermal_loss);
    CUDA_CHECK(cudaEventRecord(stop6, stream));
    // Synchronize the stop event
    CUDA_CHECK(cudaEventSynchronize(stop6));    

    CUDA_CHECK(cudaEventElapsedTime(&avg_time, start, stop1));
    CUDA_CHECK(cudaEventElapsedTime(&avg_mean_time, stop1, stop2));
    CUDA_CHECK(cudaEventElapsedTime(&a2s_time, stop2, stop3));
    CUDA_CHECK(cudaEventElapsedTime(&a2s_mean_time, stop3, stop4));
    CUDA_CHECK(cudaEventElapsedTime(&adjust_time, stop4, stop5));
    CUDA_CHECK(cudaEventElapsedTime(&thermal_time, stop5, stop6));

    std::cout << "AVG Model: " << avg_time * 0.001 << " seconds" << std::endl;
    std::cout << "AVG Sum Model: " << avg_mean_time * 0.001<< " seconds" << std::endl;
    std::cout << "A2S Model: " << a2s_time * 0.001<< " seconds" << std::endl;
    std::cout << "A2S Sum Model: " << a2s_mean_time * 0.001<< " seconds" << std::endl;
    std::cout << "Adjust Model: " << adjust_time * 0.001<< " seconds" << std::endl;
    std::cout << "Thermal Model: " << thermal_time * 0.001<< " seconds" << std::endl;

    // Verify Adjust Kernel
    
    CUDA_CHECK(cudaMemcpy(thermal_loss, d_thermal_loss, sizeof(double) * simulation_case, cudaMemcpyDeviceToHost));
    for(int i=0; i<10; ++i)
        std::cout << thermal_loss[i] << std::endl;
    
    // Clean up events and streams
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(stop3));
    CUDA_CHECK(cudaEventDestroy(stop4));
    CUDA_CHECK(cudaEventDestroy(stop5));
    CUDA_CHECK(cudaEventDestroy(stop6));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Free device memory
    CUDA_CHECK(cudaFree(d_i1_data_a2s));
    CUDA_CHECK(cudaFree(d_i2_data_a2s));
    CUDA_CHECK(cudaFree(d_vc_data_a2s));

    CUDA_CHECK(cudaFree(d_i1_data_avg));
    CUDA_CHECK(cudaFree(d_i2_data_avg));
    CUDA_CHECK(cudaFree(d_vc_data_avg));

    CUDA_CHECK(cudaFree(d_i1_mean_avg));
    CUDA_CHECK(cudaFree(d_i2_mean_avg));
    CUDA_CHECK(cudaFree(d_vc_mean_avg));

    CUDA_CHECK(cudaFree(d_i1_mean_a2s));
    CUDA_CHECK(cudaFree(d_i2_mean_a2s));
    CUDA_CHECK(cudaFree(d_vc_mean_a2s));

}
