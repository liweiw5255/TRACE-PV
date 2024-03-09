#ifndef KERNEL
#define KERNEL
#include <cmath>
#include <tuple>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <stdlib.h>
#include <chrono>
#include <stdlib.h>
#include <set>
#include <time.h>
//#include <cuda_runtime.h>
#include <cfloat>
//#include <boost/numeric/odeint.hpp>
//#include <boost/functional/hash.hpp>

using namespace std;
using namespace boost::numeric::odeint;

//define global variable
#define NUM_THREADS 5
#define N_Input 4
#define N_Output 6
#define M 500
//#define MP_sz 9 change
#define MP_sz 2
//#define mission_profile "/home/liweiw/TRACE_PV/mission_profile/ottare_pv_farm_mission_profile_5min.csv"
#define thr1 0.1
#define thr2 0.1
#define blockSize 256

// Constant
//#define Ir_ref 1000 // Irradiance reference
//#define k 1.3806e-23
//#define q 1.6022e-19
//#define T_ref 298.15 // Temperature reference
//#define T_const 273.15 
//#define pvCell_iter 10000

struct Gradient {
    double g;
    double g_d;
};

typedef vector<double> state_type;

struct signal{
    vector<double> x;
    vector<double> y;
};

typedef struct signal Signal;
typedef struct pv_cell PV_cell;


struct PVCell {
    int Ncell;
    int Nstring;
    int Nmodule;
    double Voc;
    double Isc;
    double Vmp;
    double Imp;
    double Temp_isc;
    double Temp_soc;
    double IL;
    double I0;
    double DI_factor;
    double Rsh;
    double Rs;
    int iter;
};

struct Model {

    double f;
    double RL1;
    double RL2;
    double Rc;
    double L1;
    double L2;
    double C;
    double omega = 2 * M_PI * f;
    double Vdc_ref;
    double V_g;
    double n1;
    double n2;
    double V_boost;
 
};

using namespace std;

void staticMethod(const PVCell pvCell, const Model model, const double* mpp, const double* boost, double* result, const int size);

double dynamic_single_stage(PVCell pvCell, double ir, double temp, double* PV_I_lookup);
//vector<vector<double>> dynamic_single_stage(PVCell pvCell, double ir, double temp, double* PV_I_lookup);
vector<vector<double>> dynamic_two_stage(PVCell cell, double ir, double temp, double* PV_I_lookup);

void IV_calculation(PVCell pvCell, Model model, int local_size, double* T, double* Ir, double* I, double* MPP, double* Boost);

// load data
PVCell load_pv_cell();
Model load_model();
vector<vector<double>> readCsv(string filename);

// Cluster Merging
size_t getHash(vector<double> vec);
unordered_map<size_t, vector<size_t> > clusterMerging(vector<vector<double> > MP_Input, size_t sz);
  


#endif