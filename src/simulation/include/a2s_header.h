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

#include "a2s_global_function.h"

 // Functions
void read_csv(const std::string& filename, double *& data_array);
void launchKernel(const double* S_rated, double* thermal_loss);

#endif // A2S_HEADER_H

