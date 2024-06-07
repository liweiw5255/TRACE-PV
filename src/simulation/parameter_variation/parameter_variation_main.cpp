#include "../include/a2s_parameter_variation.h"

int main() {
    int n = 1024; // Number of points
    int dimension = 2; // Number of input dimensions
    int num_outputs = 2; // Number of outputs
    std::vector<std::vector<double>> sobol_sequence = generate_sobol_sequence(n, dimension);
    
    calculate_sobol_indices(sobol_sequence, n, dimension, num_outputs);

    return 0;
}