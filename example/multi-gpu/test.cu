#include <cuda_runtime.h>
#include <iostream>

#define N 1024  // Size of the vector
#define NUM_GPUS 2  // Number of GPUs

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

int main() {
    int size = N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory and create streams and events
    float* d_A[NUM_GPUS];
    float* d_B[NUM_GPUS];
    float* d_C[NUM_GPUS];
    cudaStream_t streams[NUM_GPUS];
    cudaEvent_t startEvent[NUM_GPUS], stopEvent[NUM_GPUS];

    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMalloc(&d_A[i], size / NUM_GPUS));
        checkCudaErrors(cudaMalloc(&d_B[i], size / NUM_GPUS));
        checkCudaErrors(cudaMalloc(&d_C[i], size / NUM_GPUS));
        checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(cudaEventCreate(&startEvent[i]));
        checkCudaErrors(cudaEventCreate(&stopEvent[i]));
    }

    // Copy data to each GPU and launch kernels
    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMemcpyAsync(d_A[i], h_A + i * (N / NUM_GPUS), size / NUM_GPUS, cudaMemcpyHostToDevice, streams[i]));
        checkCudaErrors(cudaMemcpyAsync(d_B[i], h_B + i * (N / NUM_GPUS), size / NUM_GPUS, cudaMemcpyHostToDevice, streams[i]));
        checkCudaErrors(cudaEventRecord(startEvent[i], streams[i]));
        
        vectorAdd<<<(N / NUM_GPUS + 255) / 256, 256, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], N / NUM_GPUS);
        
        checkCudaErrors(cudaEventRecord(stopEvent[i], streams[i]));
        checkCudaErrors(cudaMemcpyAsync(h_C + i * (N / NUM_GPUS), d_C[i], size / NUM_GPUS, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all events to complete
    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaEventSynchronize(stopEvent[i]));
    }

    // Validate the result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            std::cerr << "Result validation failed at element " << i << "! " << h_C[i] << " != " << h_A[i] + h_B[i] << std::endl;
            exit(-1);
        }
    }
    std::cout << "Test PASSED\n";

    // Clean up
    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaFree(d_A[i]));
        checkCudaErrors(cudaFree(d_B[i]));
        checkCudaErrors(cudaFree(d_C[i]));
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cudaEventDestroy(startEvent[i]));
        checkCudaErrors(cudaEventDestroy(stopEvent[i]));
    }

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}