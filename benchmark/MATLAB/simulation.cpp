#include "simulation.h"

extern double* runSimulation(vector<double> cppData, int index) {

    std::cout << "Starting MATLAB engine..." << std::endl;

    // Start MATLAB engine
    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

    // Change directory to where the Simulink model is located
    // matlabPtr->eval(u"cd('path/to/your/simulink/model')");

    // Load the Simulink model
    matlabPtr->eval(u"load_system('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2')");

    // Set simulation parameters (if needed)
    matlabPtr->eval(u"set_param('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2', 'StopTime', '0.0266')");

    // Wait for the simulation to complete
    matlabPtr->eval(u"sim('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2')");

    // Get simulation results (if any)
    matlab::data::ArrayFactory factory;
    matlab::data::TypedArray<double> result = matlabPtr->getVariable(u"simout");

    std::cout << "Simulation results:" << std::endl;
    for (auto it = result.begin(); it != result.end(); ++it) {
        std::cout << *it << std::endl;
    }

    // Close the Simulink model without saving
    matlabPtr->eval(u"close_system('Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2', 0)");

    double *new_result= (double*)malloc(N_Output*sizeof(double));
    cout<<"Simulation No."<<index<<" Finished"<<endl;

    return new_result;

}
