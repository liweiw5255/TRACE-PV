#include "simulation.h"

extern double* runSimulation(vector<double> cppData, int index) {

    time_t start, end, t1, t2;
       
    vector<double> vec{58.6,58.6,58.6};
 
    cout<<"vector size: "<<cppData.size()<<endl;

    //cppData getHash(cppData);
    set<size_t> key_set(cppData.begin(),cppData.end());

    vector<size_t> key_vec;
    key_vec.assign(key_set.begin(),key_set.end());

    cout<<"set size: "<<key_vec.size()<<endl;
    
    cout<<"Start Simulation No."<<index<<endl; 
    
    using namespace matlab::engine;
    time (&start);
    // Connect to named shared MATLAB session started as:
    // matlab -r "matlab.engine.shareEngine('myMatlabEngine')"
    //String session(u"myMatlabEngine");
    //std::unique_ptr<MATLABEngine> matlabPtr = connectMATLAB(session);
    //std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    //FutureResult<std::unique_ptr<MATLABEngine>> matlabFuture = startMATLABAsync();
    //std::unique_ptr<MATLABEngine> matlabPtr = matlabFuture.get();

    // Create MATLAB data array factory
    matlab::data::ArrayFactory factory;
    
    // Create struct for simulation parameters
    /*matlab::data::StructArray parameterStruct = factory.createStructArray({ 1,7 }, {
        "SaveOutput",
        "OutputSaveName",
        "SaveTime",
        "TimeSaveName",
        "StartTime",
        "StopTime",
        "ReturnWorkspaceOutputs"});
        
    parameterStruct[0]["SaveOutput"] = factory.createCharArray("on");
    parameterStruct[0]["OutputSaveName"] = factory.createCharArray("yOut");
    parameterStruct[0]["SaveTime"] = factory.createCharArray("on");
    parameterStruct[0]["TimeSaveName"] = factory.createCharArray("tOut");
    parameterStruct[0]["StartTime"] = factory.createScalar(0.0);
    parameterStruct[0]["StopTime"] = factory.createScalar(10.0);
    parameterStruct[0]["ReturnWorkspaceOutputs"] = factory.createCharArray("on");
    */
    time(&t1);
    std::cout << "Initialization Time: "<< t1 - start << std::endl;
    matlabPtr->eval(u"addpath('../model_generator');");
    u16string name = u"/home/liweiw/TRACE_PV/models/AverageModel_Attempt_3_ConstV_CC.slx";
    //u16string name = u"/home/liweiw/TRACE_PV/models/test.slx";
    // Put simulation parameter struct in MATLAB
    //matlabPtr->setVariable(u"parameterStruct", parameterStruct);
    
    matlab::data::CharArray filename = factory.createCharArray(name);
    matlabPtr->setVariable(u"filename", std::move(filename)); 
    //matlab::data::TypedArray<double> mission_profile = factory.createArray<double>(vec);
        
    matlabPtr->eval(u"load_system(filename)");
    time(&t2);
    std::cout << "Load Model Time: "<< t2 - t1 << std::endl;

    //FutureResult<void> loadFuture = matlabPtr->evalAsync(u"load_system(filename)");
    //std::cout << "Loading Simulink model... " << std::endl;
    //std::future_status loadStatus;
    //do {
    //    loadStatus = loadFuture.wait_for(std::chrono::seconds(1));
    //} while (loadStatus != std::future_status::ready);
    //matlabPtr->eval(u"set_param(filename, parameterStruct);");

    
    //std::cout << "Initialization Time: "<< end_1- start << std::endl; 
    std::cout << "Running simulation... " << std::endl;
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename,'ReturnWorkspaceOutputs', 'on','StopTime',60);");
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename,'ReturnWorkspaceOutputs', 'on','StopTime',1,'AccelVerboseBuild','on');");
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename,'StartTime','0.0', 'StopTime','60.0');");
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename, parameterStruct);");
    matlabPtr->eval(u"simOut = sim(filename,'StartTime','0.0', 'StopTime','7.0');");
    
    //std::future_status simStatus;
    //do {
    //    simStatus = loadFuture.wait_for(std::chrono::seconds(1));
    //} while (simStatus != std::future_status::ready);
    std::cout << "Simulation complete\n"<<std::endl;
    
    //matlabPtr->eval(u"Vdc=simOut.yout{1}.Values.Data(end)");
    
    time (&end);
    std::cout << "Simulation Time: "<< end - t2 << std::endl; 
    std::cout << "Total Simulation Time: "<< end - start << std::endl; 
    /*
    matlabPtr->eval(u"Vdc=simOut.yout{1}.Values.Data(end)");
    matlabPtr->eval(u"Idc=simOut.yout{2}.Values.Data(end);");
    matlabPtr->eval(u"Vac=simOut.yout{3}.Values.Data(end);");
    matlabPtr->eval(u"Iac=simOut.yout{4}.Values.Data(end);");
   
    
    // Get the result from MATLAB
    matlab::data::TypedArray<double> Vdc = matlabPtr->getVariable(u"Vdc");
    matlab::data::TypedArray<double> Idc = matlabPtr->getVariable(u"Idc");
    matlab::data::TypedArray<double> Vac = matlabPtr->getVariable(u"Vac");
    matlab::data::TypedArray<double> Iac = matlabPtr->getVariable(u"Iac");

    // Display results
    std::cout << "V_dc: " <<  Vdc[0] << std::endl;
    std::cout << "I_dc: " <<  Idc[0] << std::endl;
    std::cout << "V_ac: " <<  Vac[0] << std::endl;
    std::cout << "I_ac: " <<  Iac[0] << std::endl;
 
    double *result= (double*)malloc(N_Output*sizeof(double));

    result[0]=Vdc[0];
    result[1]=Idc[0];
    result[2]=Vac[0];
    result[3]=Iac[0];
  

    cout<<"Simulation No."<<index<<" Finished"<<endl;*/

    double *result = 0;
    
    return result;  
    
}
