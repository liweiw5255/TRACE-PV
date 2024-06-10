#ifndef SIMULATION
#define SIMULATION

//include MATLAB library
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"

#include <iostream>
#include <vector>
#include <pthread.h>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <stdlib.h>
#include <set>
#include <time.h>
//#include <boost/functional/hash.hpp>

using namespace std;

//define global variable
#define NUM_THREADS 5
#define N_Input 4
#define N_Output 6
#define M 500
//#define MP_sz 9 change
#define MP_sz 2

//functions
double* runSimulation(vector<double> cppData, int index);
size_t getHash(vector<double> vec);
unordered_map<size_t, vector<size_t> > clusterMerging(vector<vector<double> > MP_Input, size_t sz);

#endif
