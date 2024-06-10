//https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/

#include "include/a2s_header.h"
#include "include/a2s_parameters.h"


double convert_S_rated(double I, double Ir){
    return 100 + I + Ir;
}

int main(int argc, char* argv[]) {
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0){
        // Print simulation time
        std::cout << "Simulation time: " << simulation_time << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();  
    
    // Define Local Variables
    int reduced_size; 
    int *local_size = new int[size];

    // Variable for Mission Profile
    double *S_rated, *local_S_rated, *local_thermal_loss;

    /********************** Part 1:  Load Mission Profile *****************************/
    
    if(rank == 0){
        size_t MP_size;
        double T_min_val, T_max_val, Ir_min_val, Ir_max_val, RH_min_val, RH_max_val;
        std::vector<size_t> key_list;
        std::vector<std::vector<double>> MP_input;

        // Load Mission Profile

        // TODO: Add Grid Information(Grid AC Voltage, Grid Frequency, Power Factor), 
        //           Environmental Information (Module Temperature, Ambient Temperature (Optional), Solar Irradiance, RH)
        
        try {
            read_csv("../mission_profile/ottare_pv_farm_mission_profile_5min.csv", MP_input);
            MP_size = MP_input.size();

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }

        /********************** Part 2:  Sobol Screening *****************************/
    
        // Variable for Sobol Sensitivity Analysis
        int n = 1024; // Number of points
        int dimension = 2; // Number of input dimensions
        int num_outputs = 2; // Number of outputs
        std::vector<std::vector<double>> sobol_sequence = generate_sobol_sequence(n, dimension);
        
        auto [S, ST] = calculate_sobol_indices(sobol_sequence, n, dimension, num_outputs);
        std::cout << "First-order Sobol indices (S): " << std::endl;
        for (int k = 0; k < num_outputs; ++k) {
            std::cout << "Output " << k + 1 << ": ";
            for (int j = 0; j < dimension; ++j) {
                std::cout << S[j][k] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Total-order Sobol indices (ST): " << std::endl;
        for (int k = 0; k < num_outputs; ++k) {
            std::cout << "Output " << k + 1 << ": ";
            for (int j = 0; j < dimension; ++j) {
                std::cout << ST[j][k] << " ";
            }
            std::cout << std::endl;
        }
   

        /********************** Part 3:  Cluster Merging *****************************/

        // Perform Cluster Merging
        std::unordered_map<size_t, std::vector<size_t> > hash_list = clusterMerging(MP_input, MP_size); 
        
        // Get the keys from the dictionary
        for(const auto iter : hash_list){
            key_list.push_back(iter.first);
        }   
        // Get the size of key list
        reduced_size = key_list.size();
        S_rated = new double[reduced_size];

        int element_per_processor = std::ceil(reduced_size/size);
        
        for(int i = 0; i < size - 1 ; ++i)
            local_size[i] = element_per_processor;
        local_size[size-1] = std::max(element_per_processor,reduced_size - (size-1) * element_per_processor);

        for(int i=0; i<reduced_size; ++i){
            double T = (MP_input[hash_list[key_list[i]][0]][1] - 32 )*5/9;
            double Ir = MP_input[hash_list[key_list[i]][0]][0]; 
            S_rated[i] = convert_S_rated(T, Ir);
        }

    }

    // Wait for the missio profile load finish
    MPI_Barrier(comm);

    // Broadcast mission profile size to each processor
    MPI_Bcast(local_size, size, MPI_INT, 0, comm);

    // Initialize the local mission profile array
    local_S_rated = new double[local_size[rank]];
    local_thermal_loss = new double[local_size[rank]];

    std::cout << "rank = " << rank << " " << local_size[rank] << std::endl;

    // Scatter 
    MPI_Scatter(S_rated, *local_size, MPI_DOUBLE, local_S_rated, *local_size, MPI_DOUBLE, 0, comm);

    /********************** Part 4:  A2S Model and Thermal Model Simulation *****************************/
        
    // Perform A2S Model Simulation
    for(int j=0; j<local_size[rank]; j+=batchSize){
        int index = 0;
        for(int k=0; k<rank; ++k)
            index += local_size[k];
        launchKernel(local_S_rated + j, local_size[rank], index + j, local_thermal_loss + j, comm);   
    }

       
    /********************** Part 5:  Critical Degradation Condition Check *****************************/

    if(rank == 0){
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate duration
        std::chrono::duration<double> duration = end - start;

        // Output the duration in seconds
        std::cout << "Elapsed time: " << duration.count() << " seconds" << std::endl;
    }
  
    MPI_Finalize();

    return 0;
}

