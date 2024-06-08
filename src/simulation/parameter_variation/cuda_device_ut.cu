#include "../include/a2s_device_function.h"

__device__ void rungeKuttaSolve(const double* initialConditions, const double t0, const double tf, double h, const double* params, double* result) {

    // For Adaptive Solver
    const double adaptive_a[6][5] = {
        {0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0/4.0, 0.0, 0.0, 0.0, 0.0},
        {3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0},
        {1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0},
        {439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0},
        {-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0}
    }; // Coefficients for RK45 method

    const double adaptive_b1[6] = {16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0}; // Coefficients for primary solution
    // const double adaptive_b2[6] = {25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0};// Coefficients for secondary solution
    const double adaptive_c[6] = {0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0,1.0,1.0/2.0}; // Coefficients for intermediate steps
  
    double yn[DIM], y_primary[DIM];
    // double y_secondary[DIM], error_norm = 0.0, yn_norm = 0.0, tolerance;

    // Compute new step size for the next iteration
    // double safety_factor = 0.9, h_new, h_max, h_min;
    // int num_steps = int((tf - t0)/h) + 1; 
    int index = 0, i;
    double t = t0;
    
    for(i = 0; i < DIM; ++i) yn[i] = initialConditions[i];
    double k1[DIM], k2[DIM], k3[DIM], k4[DIM], k5[DIM], k6[DIM], ynTemp[DIM];
 
    while (t < tf ) {

        // Copy yn to result array
        for(i = 0; i < DIM; ++i) result[index*DIM+i] = yn[i];
        // memcpy(result + index * DIM, yn, DIM * sizeof(double));

        // Calculate K1
        fun(yn, t, params, k1);

        // Calculate K2
        for (i = 0; i < DIM; i++) ynTemp[i] = yn[i] + k1[i] * adaptive_a[1][0] * h;
        fun(ynTemp, t + adaptive_c[1] * h, params, k2);

        // Calculate K3
        for (i = 0; i < DIM; i++) ynTemp[i] = yn[i] + (k1[i] * adaptive_a[2][0] + k2[i] * adaptive_a[2][1]) * h;   
        fun(ynTemp, t + adaptive_c[2] * h, params, k3);
           
        // Calculate K4
        for (i = 0; i < DIM; i++) ynTemp[i] = yn[i] + (k1[i] * adaptive_a[3][0] + k2[i] * adaptive_a[3][1] + k3[i] * adaptive_a[3][2]) * h;
        fun(ynTemp, t + adaptive_c[3] * h, params, k4);
     
        // Calculate K5
        for (i = 0; i < DIM; i++) ynTemp[i] = yn[i] + (k1[i] * adaptive_a[4][0] + k2[i] * adaptive_a[4][1] + k3[i] * adaptive_a[4][2] + k4[i] * adaptive_a[4][3]) * h;
        fun(ynTemp, t + adaptive_c[4] * h, params, k5);

        // Calculate K6
        for (i = 0; i < DIM; i++) ynTemp[i] = yn[i] + (k1[i] * adaptive_a[5][0] + k2[i] * adaptive_a[5][1] + k3[i] * adaptive_a[5][2] + k4[i] * adaptive_a[5][3] + k5[i] * adaptive_a[5][4]) * h;
        fun(ynTemp, t + adaptive_c[5] * h, params, k6);
        
        for (i = 0; i < DIM; i++) y_primary[i] = yn[i] + (k1[i] * adaptive_b1[0] + k2[i] * adaptive_b1[1] + k3[i] * adaptive_b1[2] + k4[i] * adaptive_b1[3] + k5[i] * adaptive_b1[4] + k6[i] * adaptive_b1[5]) * h;
        // for (i = 0; i < DIM; i++) y_secondary[i] = yn[i] + (k1[i] * adaptive_b2[0] + k2[i] * adaptive_b2[1] + k3[i] * adaptive_b2[2] + k4[i] * adaptive_b2[3] + k5[i] * adaptive_b2[4] + k6[i] * adaptive_b2[5]) * h;  
   
        /*
        for (i = 0; i < DIM; i++){
            yn_norm = yn[i] * yn[i];   
            error_norm += (y_primary[i] - y_secondary[i]) * (y_primary[i] - y_secondary[i]);
        } 
        
        // Estimate error
        error_norm = sqrtf(error_norm);

        // Compute the norm of the current solution
        yn_norm = sqrtf(yn_norm);
         
        // Check against combined tolerance
        tolerance = max(abs_tol, rel_tol * yn_norm);
 
        // Adjust step size based on error
        if (error_norm <= tolerance) {
             
            // Accept the step
            for (i = 0; i < DIM; i++)  yn[i] = y_primary[i];
            t += h;
      
            // Ensure we capture the final value exactly at tf
            if (t > tf) {
                t = tf;
             
                // Copy yn to result array
                memcpy(result + index * DIM, yn, DIM * sizeof(double));
                index++;
                break;
            } // if (t > tf)
            
            index++;

        } // if (error_norm <= tolerance)

        h_new = h * safety_factor * pow(tolerance / error_norm, 0.25);

        // Ensure the new step size is not too large or too small
        h_max = 2 * h;
        h_min = h / 2; 

        h = min(h_max, max(h_new, h_min));

        // Ensure h stays within the bounds to avoid infinite loop
        h = min(h, (tf - t) / (num_steps - index - 1));  
  */

        for(i = 0; i < DIM; ++i) yn[i] = y_primary[i];
        t += h;
        index++;

    } // while (t < tf)
   
} // Function End
