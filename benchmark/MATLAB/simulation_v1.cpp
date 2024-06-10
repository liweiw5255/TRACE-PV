#include "simulation.h"

extern double* runSimulation(vector<double> cppData, int index) {

    time_t start,end, start_1, end_1, start_2, end_2;
    time (&start);


    cout<<"vector size: "<<cppData.size()<<endl;

    //cppData getHash(cppData);
    set<size_t> key_set(cppData.begin(),cppData.end());

    vector<size_t> key_vec;
    key_vec.assign(key_set.begin(),key_set.end());

    cout<<"set size: "<<key_vec.size()<<endl;
    
    cout<<"Start Simulation No."<<index<<endl; 
    
    using namespace matlab::engine;
   
    // Connect to named shared MATLAB session started as:
    // matlab -r "matlab.engine.shareEngine('myMatlabEngine')"
    //String session(u"myMatlabEngine");
    //std::unique_ptr<MATLABEngine> matlabPtr = connectMATLAB(session);
    //std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    FutureResult<std::unique_ptr<MATLABEngine>> matlabFuture = startMATLABAsync();
    std::unique_ptr<MATLABEngine> matlabPtr = matlabFuture.get();

    // Create MATLAB data array factory
    matlab::data::ArrayFactory factory;
    
    // Create struct for simulation parameters
    matlab::data::StructArray parameterStruct = factory.createStructArray({ 1,7 }, {
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

    matlabPtr->eval(u"addpath('../model_generator');");
    u16string name = u"/home/liweiw/TRACE_PV/models/Generated_Model.slx";
    //u16string name = u"/home/liweiw/TRACE_PV/models/test.slx";
    // Put simulation parameter struct in MATLAB
    matlabPtr->setVariable(u"parameterStruct", parameterStruct);
    
    matlab::data::CharArray filename = factory.createCharArray(name);
    matlabPtr->setVariable(u"filename", std::move(filename)); 
    
    time (&end_1);
    
    std::cout << "Initialiation Time: "<< end_1- start << std::endl; 
   
    time (&start_1);

    FutureResult<void> loadFuture = matlabPtr->evalAsync(u"load_system(filename)");
    //std::cout << "Loading Simulink model... " << std::endl;
    std::future_status loadStatus;
    do {
        loadStatus = loadFuture.wait_for(std::chrono::seconds(1));
    } while (loadStatus != std::future_status::ready);
    //matlabPtr->eval(u"set_param(filename, parameterStruct);");

    
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename,'ReturnWorkspaceOutputs', 'on','StopTime',1,'AccelVerboseBuild','on');");
    //FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename,parameterStruct, 'AccelVerboseBuild','on');");
    FutureResult<void> simFuture = matlabPtr->evalAsync(u"simOut = sim(filename);");
    std::cout << "Running simulation... " << std::endl;
    std::future_status simStatus;
    do {
        simStatus = loadFuture.wait_for(std::chrono::seconds(1));
    } while (simStatus != std::future_status::ready);
    std::cout << "Simulation complete\n"<<std::endl;
   
    time (&end_2);
    std::cout << "Simulation Time: "<< end_2 - start_1 << std::endl; 
    
    time (&start_2);
    matlabPtr->eval(u"i_load=simOut.out_i_load.Data(end);");
    matlabPtr->eval(u"v_load=simOut.out_v_load.Data(end);");
    matlabPtr->eval(u"T=simOut.out_T.Data(end);");
    matlabPtr->eval(u"Vc_abc=simOut.out_Vc_abc.Data(end);");
    matlabPtr->eval(u"Vpcc_abc=simOut.out_Vpcc_abc.Data(end);");
    matlabPtr->eval(u"Vgrid_abc=simOut.out_Vgrid_abc.Data(end);");
    
    // Get the result from MATLAB
    matlab::data::TypedArray<double> i_load = matlabPtr->getVariable(u"i_load");
    matlab::data::TypedArray<double> v_load = matlabPtr->getVariable(u"v_load");
    matlab::data::TypedArray<double> T = matlabPtr->getVariable(u"T"); 
    matlab::data::TypedArray<double> Vc_abc = matlabPtr->getVariable(u"Vc_abc"); 
    matlab::data::TypedArray<double> Vpcc_abc = matlabPtr->getVariable(u"Vpcc_abc"); 
    matlab::data::TypedArray<double> Vgrid_abc = matlabPtr->getVariable(u"Vgrid_abc");
    
    // Display results
    std::cout << "i_load: " << i_load[0] << std::endl;
    std::cout << "v_load: " << v_load[0] << std::endl;
    std::cout << "T: " << T[0] << std::endl;
    std::cout << "Vc_abc: " << Vc_abc[0] << std::endl;
    std::cout << "Vpcc_abc: " << Vpcc_abc[0] << std::endl;
    std::cout << "Vgrid_abc: " << Vgrid_abc[0] << std::endl;

    time (&end);
    std::cout << "Data Transfer Time: "<< end - start_2 << std::endl; 
    std::cout << "Total Simulation Time: "<< end - start << std::endl; 

    double *result= (double*)malloc(N_Output*sizeof(double));

    result[0]=i_load[0];
    result[1]=v_load[0];
    result[2]=T[0];
    result[3]=Vc_abc[0];
    result[4]=Vpcc_abc[0];
    result[5]=Vgrid_abc[0];

    cout<<"Simulation No."<<index<<" Finished"<<endl;

    return result;

}
