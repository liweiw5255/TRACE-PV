#include "../include/a2s_device_function.h"

/*

% State Space for Two-Level Inverter Considering LCL Output Filter and Grid Connection

% Inputs: 
%   - t: Time. Define as symbolic variable when using this function.
%   - x: State. Define as symbolic variable when using this function.
%   - L1: Inductance of first stage inductor of LCL filter.
%   - R1: Resistance of first stage inductor of LCL filter.
%   - C: Capacitance of capacitor of LCL filter.
%   - L2: Inductance of second stage inductor of LCL filter.
%   - R2: Resistance of second stage inductor of LCL filter.
%   - LG: Inductance of grid referred to the primary side of transformer.
%   - RG: Resistance of grid referred to the primary side of transformer.
%   - V_phase: Instantaneous phase voltage at output of inverter.
%   - VG_phase: Instantaneous phase voltage of grid.


% Outputs:
%   - dx: Derivative of state x.
*/

__device__ void fun(const double *x, double t, const double* params, double* result){

    double VG_phase = params[0];
    double V_phase = params[1];

    double A[DIM * DIM] = {
        -(R1+RC)/L1, RC/L1, -1/L1,
        RC/(L2+LG), -(RC+R2+RG)/(L2+LG), 1/(L2+LG),
        1/C, -1/C, 0.0
    };

    double B[DIM] = {
        V_phase * (1/L1),
        VG_phase * -1/(L2+LG),
        0.0 // Assuming the third element of B is always zero
    };

    // Perform matrix-vector multiplication A*x + B
    for (int i = 0; i < DIM; ++i) {
        result[i] = B[i];
        for (int j = 0; j < DIM; ++j) {
            result[i] += A[i * DIM + j] * x[j];
        }
    }
}