//https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

#include "include/a2s_header.h"
#include "include/a2s_parameters.h"

int main(int argc, char* argv[]) {

    // Print simulation time
    std::cout << "Simulation time: " << simulation_time << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Variable for Mission Profile
    double *S_rated = new double[simulation_case];
    
    // Variable for Sobol Sensitivity Analysis
    int n = 1024; // Number of points
    int dimension = 2; // Number of input dimensions
    int num_outputs = 2; // Number of outputs
    std::vector<std::vector<double>> sobol_sequence = generate_sobol_sequence(n, dimension);
    
    // Variable for Thermal Model
    double *thermal_loss = new double[simulation_case];

    /********************** Part 1:  Load Mission Profile *****************************/
    for(int i = 0; i < simulation_case; ++i) {
        S_rated[i] = 100;
    }

    /********************** Part 2:  Cluster Merging *****************************/


    /********************** Part 3:  Sobol Method *****************************/
    
    auto [S, ST] = calculate_sobol_indices(sobol_sequence, n, dimension, num_outputs);
    std::cout << "First-order Sobol indices (S): " << std::endl;
    for (int k = 0; k < num_outputs; ++k) {
        std::cout << "Output " << k + 1 << ": ";
        for (int j = 0; j < dimension; ++j) {
            std::cout << S[j][k] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Total-order Sobol indices (ST): " << std::endl;
    for (int k = 0; k < num_outputs; ++k) {
        std::cout << "Output " << k + 1 << ": ";
        for (int j = 0; j < dimension; ++j) {
            std::cout << ST[j][k] << " ";
        }
        std::cout << std::endl;
    }

    /********************** Part 4:  A2S Model and Thermal Model Simulation *****************************/
        
    // Perform A2S Model Simulation
    launchKernel(S_rated, thermal_loss);

    /********************** Part 4:  Critical Degradation Condition Check *****************************/

    

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;

    // Output the duration in seconds
    std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
   
    return 0;
}

