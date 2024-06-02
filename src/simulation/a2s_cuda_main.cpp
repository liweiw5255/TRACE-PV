//https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

#include "include/a2s_header.h"
#include "include/a2s_parameters.h"

int main(int argc, char* argv[]) {

    // Print simulation time
    std::cout << "Simulation time: " << simulation_time << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    double *S_rated = new double[simulation_case];
    double *thermal_loss = new double[simulation_case];
    for(int i = 0; i < simulation_case; ++i) {
        S_rated[i] = 100;
    }

    // Perform matrix multiplication
    launchKernel(S_rated, thermal_loss);

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;

    // Output the duration in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
   
    return 0;
}

