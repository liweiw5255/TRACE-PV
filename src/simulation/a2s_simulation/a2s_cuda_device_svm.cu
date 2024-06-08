#include "../include/a2s_device_function.h"

__device__ void matrixMult(const double *lhs_array, const int *rhs_array, double *result, const int m, const int n, const int p){
    for(int i=0; i<m; ++i){
        for(int j=0; j<p; ++j){
            result[i*p+j] = 0.0;
            for(int k=0; k<n; ++k){
                result[i * p + j] += lhs_array[i * n + k] * rhs_array[k * p + j];
            }
        }
    }
}

__device__ void getSVM3Phase(const double* Vr_ll, double duration[duration_sz], double outputStates[duration_sz*DIM]){

   
    int i, currS;
  
    double Vr_aby_12[2], Vsv_aby[16], V1_aby[2], V2_aby[2];
    double sz1[DIM], sz2[DIM], s1[DIM], s2[DIM];
    double th1_aby, dz=0.0, d1=0.0, d2=0.0;
    
    const double Taby[9] = {sqrtf(2.0/3), -sqrtf(2.0/3)/2, -sqrtf(2.0/3)/2, 0, sqrtf(2.0)/2, -sqrtf(2.0)/2, sqrtf(2.0/3)/2, sqrtf(2.0/3)/2, sqrtf(2.0/3)/2};
    const int states[24] = {1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0}; 
    const int Vsv_ll[24] = {0, Vdc, 0, -Vdc, -Vdc, 0, Vdc, 0, 0, 0, Vdc, Vdc, 0, -Vdc, -Vdc, 0, 0, -Vdc, -Vdc, 0, Vdc, Vdc, 0, 0};
    const int sectors[12] = {2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2};
 
    Vr_aby_12[0] = Taby[0*DIM]*Vr_ll[0] + Taby[0*DIM+1]*Vr_ll[1] + Taby[0*DIM+2]*Vr_ll[2];
    Vr_aby_12[1] = Taby[1*DIM]*Vr_ll[0] + Taby[1*DIM+1]*Vr_ll[1] + Taby[1*DIM+2]*Vr_ll[2];
    
    matrixMult(Taby, Vsv_ll, Vsv_aby, DIM, DIM, 8);

    // Calculate angle th1_aby
    if (Vr_aby_12[0] == 0 && Vr_aby_12[1] == 0) {
        th1_aby = 0;
    } else {
        th1_aby = atan(Vr_aby_12[1] / Vr_aby_12[0]) * 180.0 / M_PI; // Convert radians to degrees
        if (Vr_aby_12[0] >= 0 && Vr_aby_12[1] >= 0) {
            th1_aby += 0;
        } else if (Vr_aby_12[0] < 0 && Vr_aby_12[1] >= 0) {
            th1_aby += 180;
        } else if (Vr_aby_12[0] < 0 && Vr_aby_12[1] < 0) {
            th1_aby += 180;
        } else {
            th1_aby += 360;
        }
    }

    // Select sector based on angle
    if (th1_aby >= 30.0 && th1_aby < 90.0) {
        currS = 1;
    } else if (th1_aby >= 90.0 && th1_aby < 150.0) {
        currS = 2;
    } else if (th1_aby >= 150.0 && th1_aby < 210.0) {
        currS = 3;
    } else if (th1_aby >= 210.0 && th1_aby < 270.0) {
        currS = 4;
    } else if (th1_aby >= 270.0 && th1_aby < 330.0) {
        currS = 5;
    } else {
        currS = 6;
    }

    if(currS == 1 || currS == 3 || currS == 5){
        V1_aby[0] = Vsv_aby[sectors[2*currS - 2]-1];
        V1_aby[1] = Vsv_aby[sectors[2*currS - 2]-1 + 8];
        V2_aby[0] = Vsv_aby[sectors[2*currS - 1]-1];
        V2_aby[1] = Vsv_aby[sectors[2*currS - 1]-1 + 8];
        
        s1[0] = states[sectors[2*currS - 2]-1];
        s1[1] = states[sectors[2*currS - 2]-1 + 8];
        s1[2] = states[sectors[2*currS - 2]-1 + 16];

        s2[0] = states[sectors[2*currS - 1]-1];
        s2[1] = states[sectors[2*currS - 1]-1 + 8];
        s2[2] = states[sectors[2*currS - 1]-1 + 16];
    }
    else{
        V1_aby[0] = Vsv_aby[sectors[2*currS - 1]-1];
        V1_aby[1] = Vsv_aby[sectors[2*currS - 1]-1 + 8];
        V2_aby[0] = Vsv_aby[sectors[2*currS - 2]-1];
        V2_aby[1] = Vsv_aby[sectors[2*currS - 2]-1 + 8];

        s1[0] = states[sectors[2*currS - 1]-1];
        s1[1] = states[sectors[2*currS - 1]-1 + 8];
        s1[2] = states[sectors[2*currS - 1]-1 + 16];

        s2[0] = states[sectors[2*currS - 2]-1];
        s2[1] = states[sectors[2*currS - 2]-1 + 8];
        s2[2] = states[sectors[2*currS - 2]-1 + 16];
    }
        
    d1 = (V2_aby[1] * Vr_aby_12[0] - V2_aby[0]*Vr_aby_12[1])/(V1_aby[0]*V2_aby[1] - V1_aby[1]*V2_aby[0]);
    d2 = (-V1_aby[1] * Vr_aby_12[0] - V1_aby[0]*Vr_aby_12[1])/(V1_aby[0]*V2_aby[1] - V1_aby[1]*V2_aby[0]);    
    dz = 1 - d1 - d2;
   
    duration[0] = switching_period*dz/4;
    duration[1] = switching_period*d1/2;
    duration[2] = switching_period*d2/2;
    duration[3] = switching_period*dz/2;
    duration[4] = switching_period*d2/2;
    duration[5] = switching_period*d1/2;
    duration[6] = switching_period*dz/4;
       
    /*
    double duration_sum = 0.0;
  
    double new_duration[duration_sz]{0.0};
    for(i=0; i<int(duration_sz/2); ++i){
        double total_state_time = duration[i] + duration[duration_sz-1-i];
        new_duration[i] = std::floor(duration[i]/step_size) * step_size;
        if(std::round(total_state_time/step_size)*step_size > total_state_time){   
            new_duration[duration_sz-1-i] = std::ceil(duration[duration_sz-1-i]/step_size) * step_size;
        }
        else{
            new_duration[duration_sz-1-i] = std::floor(duration[duration_sz-1-i]/step_size) * step_size;
        }
        duration_sum += duration[i] + duration[7-1-i];
    }
    new_duration[3] = switching_period - duration_sum;
    duration_sum += new_duration[3];

    for(i=0; i<duration_sz; ++i){
        duration[i] = new_duration[i];
    }
    */
   
    for(i=0; i<DIM; ++i){
        sz1[i] = 0;
        sz2[i] = 1;
    }
    
    for(i=0; i<DIM; ++i){
        outputStates[0*7+i] = sz1[i];
        outputStates[1*7+i] = s1[i];
        outputStates[2*7+i] = s2[i];
        outputStates[3*7+i] = sz2[i];
        outputStates[4*7+i] = s2[i];
        outputStates[5*7+i] = s1[i];
        outputStates[6*7+i] = sz1[i];
    }
    
    delete[] Vr_aby_12;
    delete[] V1_aby;
    delete[] V2_aby;
}