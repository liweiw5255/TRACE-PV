// multi_gpu.cu
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel(int gpu_id) {
    // Simple kernel to run on each GPU
    printf("Hello from GPU %d\n", gpu_id);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the number of GPUs
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    checkCUDAError("cudaGetDeviceCount");

    if (rank < deviceCount) {
        // Assign GPU to each process
        cudaSetDevice(rank);
        checkCUDAError("cudaSetDevice");

        // Launch kernel
        kernel<<<1, 1>>>(rank);
        checkCUDAError("kernel launch");

        cudaDeviceSynchronize();
        checkCUDAError("cudaDeviceSynchronize");
    } else {
        std::cerr << "Rank " << rank << " exceeds device count " << deviceCount << std::endl;
    }

    MPI_Finalize();
    return 0;
}