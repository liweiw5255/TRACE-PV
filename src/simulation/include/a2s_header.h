#ifndef A2S_HEADER_H
#define A2S_HEADER_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <chrono>
#include <functional>
#include <mpi.h>
#include <unordered_map>

// Define constants
const int MAX_BITS = 30;
const int DIMENSION = 2; // Adjust the dimension as needed

#include "a2s_global_function.h"

 // Functions
void read_csv(const std::string& filename, std::vector<std::vector<double>>& data);
void find_min_max(const double* data, int size, double& minVal, double& maxVal);
std::unordered_map<size_t, std::vector<size_t> > clusterMerging(std::vector<std::vector<double> > MP_Input, size_t sz);


// Launch CUDA Kernel
void launchKernel(const double* S_rated, const int local_simulation_case, const int offset, double* thermal_loss, MPI_Comm comm);

std::vector<std::vector<double>> generate_sobol_sequence(int n, int dimension);
  
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> calculate_sobol_indices(const std::vector<std::vector<double>>& sobol_sequence, int n, int dimension, int num_outputs);

#endif // A2S_HEADER_H

