#pragma once
#include "../include/a2s_parameters.h"
#include <cuda_runtime_api.h>

// typedef void (*ODEFunc)(const double*, double, const double*, double*);
__device__ void fun(const double *x, double t, const double* params, double* result);
// __device__ void rungeKuttaSolve(ODEFunc f,const double* initialConditions, double t0, double tf, double h, const double* params, double* result);
__device__ void rungeKuttaSolve(const double* initialConditions, const double t0, const double tf, double h, const double* params, double* result);
__device__ void getSVM3Phase(const double* Vr_ll, double duration[duration_sz], double outputStates[duration_sz*DIM]);
__device__ double doubleAtomicAdd(double* address, double val);