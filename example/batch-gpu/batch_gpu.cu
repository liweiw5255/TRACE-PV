#include <cuda_runtime.h>
#include <iostream>

// Kernel to process the data
__global__ void processData(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;  // Example operation
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    size_t totalSize = 1e5 * 16e3 / 60 * 1e3 * 3;  // Total size of the dataset
    size_t batchSize = 16e3 / 60 * 1e3 * 3 * 6 * 24 * 12;   // Size of each batch

    std::cout << batchSize * sizeof(double) / (1024 * 1024 * 1024) << std::endl;

    float *h_data = new float[batchSize];
    float *d_data;
    checkCudaError(cudaMalloc((void**)&d_data, batchSize * sizeof(float)), "Failed to allocate device memory");

    for (int i = 0; i < totalSize; i += batchSize) {
        int currentBatchSize = std::min(batchSize, totalSize - i);

        // Initialize host data for the current batch
        for (int j = 0; j < currentBatchSize; ++j) {
            h_data[j] = static_cast<float>(i + j);
        }

        // Copy data to device
        checkCudaError(cudaMemcpy(d_data, h_data, currentBatchSize * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to device");

        // Define grid and block sizes
        int threadsPerBlock = 256;
        int blocksPerGrid = (currentBatchSize + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel
        processData<<<blocksPerGrid, threadsPerBlock>>>(d_data, currentBatchSize);
        checkCudaError(cudaGetLastError(), "Kernel launch failed");
        checkCudaError(cudaDeviceSynchronize(), "Failed to synchronize");

        // Copy processed data back to host
        checkCudaError(cudaMemcpy(h_data, d_data, currentBatchSize * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy data to host");

        // Process the results on the host if needed
        // ...
    }

    // Clean up
    delete[] h_data;
    checkCudaError(cudaFree(d_data), "Failed to free device memory");

    return 0;
}