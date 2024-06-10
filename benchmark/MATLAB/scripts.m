mission_profile = readtable('../../mission_profile/ottare_pv_farm_mission_profile_5min.csv');

mission_profile = mission_profile(1:49793,:);

s_rated = mission_profile.Var1 + mission_profile.Var2 + mission_profile.Var3;

tic;
sim("../../model/Two_Level_SVM_No_PV_No_Control_Sw_Benchmark_v2.slx", 'StopTime','0.0266');
toc