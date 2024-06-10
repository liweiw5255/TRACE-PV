#include "../include/tracepv_header.h"
#include "../include/tracepv_parameters.h"
#include "../include/tracepv_data.h"

// TODO: Apply User Input

int main(int argc, char* argv[]) {

    size_t key_size, MP_size;
    double T_min_val, T_max_val, Ir_min_val, Ir_max_val, RH_min_val, RH_max_val;
    std::vector<size_t> key_list;
    std::vector<std::vector<double>> MP_input;

     // Load Mission Profile

    // TODO: Add Grid Information(Grid AC Voltage, Grid Frequency, Power Factor), 
    //           Environmental Information (Module Temperature, Ambient Temperature (Optional), Solar Irradiance, RH)
    
    try {
        read_csv("mission_profile.csv", MP_input);
        MP_size = MP_input.size();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Perform Cluster Merging
    std::unordered_map<size_t, std::vector<size_t> > hash_list = clusterMerging(MP_input, MP_size); 
    
    // Get the keys from the dictionary
    for(const auto iter : hash_list){
        key_list.push_back(iter.first);
    }   
    // Get the size of key list
    key_size = key_list.size();

    // Initialize Mission Profile Array
    double *T = new double[key_size];
    double *Ir = new double[key_size];
    double *RH = new double[key_size];

    
    // Load Temperature and Irradiance input
    for(size_t i=0; i<key_size; ++i){
        T[i] = (MP_input[hash_list[key_list[i]][0]][1] - 32 )*5/9;
        Ir[i] = MP_input[hash_list[key_list[i]][0]][0];        
        // Ir, T, Grid Voltage, Power Factor

        // TODO: Make the threshold as a variable in the parameter head file -> FIXED
    } 

    find_min_max(T, MP_size, T_min_val, T_max_val);
    find_min_max(Ir, MP_size, Ir_min_val, Ir_max_val);
    //find_min_max(RH, MP_size, RH_min_val, RH_max_val);

    double *i1_data_a2s = new double[simulation_size * DIM];
    double *i2_data_a2s = new double[simulation_size * DIM];
    double *vc_data_a2s = new double[simulation_size * DIM];

    launch_cuda(T, Ir, key_size, i1_data_a2s, i2_data_a2s, vc_data_a2s);
    
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.

    return 0;
}
