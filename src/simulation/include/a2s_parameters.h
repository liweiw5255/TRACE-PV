#include "a2s_header.h"

#define DIM 3
#define duration_sz 7
#define Vmag 285.7229 * sqrtf(2) * sqrtf(3)
#define VG_mag 480.0 * sqrtf(2) / sqrtf(3)
#define V_g 1.5e4 / sqrtf(3.0)
#define abs_tol 1e-7
#define rel_tol 1e-3
#define BLOCK_SIZE 512
#define MP_sz 3


const double switching_frequency = 10e3;
const double switching_period = 1.0 / switching_frequency;
const double simulation_time = 266 * switching_period;
// const size_t simulation_case = 6e4;
const size_t batchSize = 5e3;
const double threshold[MP_sz]{0.1};

// Average Model Simulation Parameters
const int Ncell = 96;
const int Nstring = 29;
const int Nmodule = 19;
const double Voc = 64.2;
const double Isc = 5.96;
const double Vmp = 54.7;
const double Imp = 5.58;
const double Temp_isc = 0.061745;
const double Temp_soc = -0.2727;
const double IL = 5.9657;
const double I0 = 6.3076e-12;
const double DI_factor = 0.94489;
const double Rsh = 393.2054;
const double Rs = 0.37428;
const int iter = 1000;

const int f = switching_frequency;
const double RL1 = 0.0198;
const double RL2 = 0.0053;
const double Rc = 10;
// const double L1 = 3.5018e-4;
// const double L2 = 9.3183e-5;
// const double C = 9.7860e-5;
// const double V_g = 1.5e4 / sqrt(3.0);
const double n1 = 1.5e4;
const double n2 = 480;
const double Vdc_ref = 1000;
const double omega = 2 * M_PI * f;

// Average Model Size
const int avg_size = 5e5;
const double step_size = 1e-7;
const int simulation_size = int(simulation_time / switching_period);
const int duration_size = int(switching_period / step_size);
// const int simulation_total_size = simulation_case * simulation_size;

// Define DC-link voltage. Can put in loop later if it varies with time.
const int Vdc = 1000;

// Define grid voltage phase peak magnitude, frequency, and phase referred to primary side of transformer
const int VG_freq = 60;
const int VG_phase = 0;

//const double simulation_time = 0.01;

// const int f = 60;
const double PhaseShift = 0.1974 + M_PI/6 + 0.02;  // radians

//const double searching_size = 1e-2;
const double searching_size = 1e-2;
const double searching_tolerance = 1e-2;

// simulatoin model parameters
const double L1 = 3e-4;
const double R1 = 0.03;
const double C = 9.7860e-05;
const double RC = 1.0;
const double L2 = 9.3183e-05;
const double R2 = 0.0053;
const double LG = 0.0;
const double RG = 0.0;

const double safetyFactor = 0.9;
const double min_step = 0.0;
const double max_step = 1.0;

const std::string avg_data_path = "/home/liweiw/New_TRACE_PV/data/";

