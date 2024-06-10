#include <mpi.h>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace matlab::engine;
using namespace matlab::data;

void runSimulinkModel(std::unique_ptr<MATLABEngine>& matlabPtr, int rank, const std::vector<double>& params) {
    // Set simulation parameters based on rank and provided parameters
    std::string set_param_command = "set_param('MATLAB/Two_Level_SVM_No_PV_No_Control_Average_Benchmark_v2', 'StopTime', '" + std::to_string(params[0]) + "')";
    matlabPtr->eval(convertUTF8StringToUTF16String(set_param_command));

    // Start the simulation
    matlabPtr->eval(u"set_param('MATLAB/Two_Level_SVM_No_PV_No_Control_Average_Benchmark_v2', 'SimulationCommand', 'start')");

    // Wait for the simulation to complete
    matlabPtr->eval(u"sim('MATLAB/Two_Level_SVM_No_PV_No_Control_Average_Benchmark_v2')");

    // Get simulation results (assuming results are stored in 'simout')
    TypedArray<double> result = matlabPtr->getVariable(u"simout");

    std::cout << "Rank " << rank << " simulation results:" << std::endl;
    for (auto it = result.begin(); it != result.end(); ++it) {
        std::cout << *it << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialize MATLAB engine once
    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

    // Change directory to where the Simulink model is located
    matlabPtr->eval(u"cd('/home/liweiw/TRACE_PV/new_src/TRACE-PV/benchmark/MATLAB/')");

    // Load the Simulink model once
    matlabPtr->eval(u"load_system('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2')");

    // Define parameters for each rank
    std::vector<std::vector<double>> parameters = {
        {10.0},  // Parameters for rank 0
        {20.0},  // Parameters for rank 1
        {30.0},  // Parameters for rank 2
        {40.0}   // Parameters for rank 3
    };

    // Run simulation for the current rank with its parameters
    runSimulinkModel(matlabPtr, world_rank, parameters[world_rank]);

    // Close the Simulink model without saving at the end
    if (world_rank == 0) {
        matlabPtr->eval(u"close_system('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2', 0)");
    }

    MPI_Finalize();
    return 0;
}