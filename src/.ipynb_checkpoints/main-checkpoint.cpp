/#include "simulation.h"
//install_name_tool -add_rpath $matlabroot/extern/bin/maci64/ main   

vector<vector<double> > readCsv(string filename, int rank, int size){
    
    vector< vector<double> > MP_Input(1,vector<double>(MP_sz));    
    
    ifstream fin;
    fin.open(filename);

    if(!fin){
        cout<<"Wrong filename!"<<endl;
        exit(-1);
    }
   
    string line;
    getline(fin,line);
    while(getline(fin,line)){
        vector<double> row;
        string number;
        istringstream lineStream(line);
        string::size_type sz;
        while(getline(lineStream,number,',')){
            row.push_back(stof(number, &sz));
        }
        MP_Input.push_back(row);
    }
    return MP_Input; 
}


void MonteCarlo(vector<vector<double> > input_parameters,vector<vector<double> > random_parameters){
    
    // generate input parameters 
    std::default_random_engine generator;
    // 0: seed(100)
    // 1: seed(200)
    // 2: seed(300)
    // 3: seed(400)

    // Another set
    // 0: seed(500)
    // 1: seed(600)
    // 2: seed(700)
    // 3: seed(800)

    generator.seed(100);
    //std::normal_distribution<double> distribution;
    double number;
    
    for(int i=0; i<4; ++i){
        double upper_bound = input_parameters[i][0] + input_parameters[i][1];
        double lower_bound = input_parameters[i][0] - input_parameters[i][1];
        std::normal_distribution<double>distribution(input_parameters[i][0],input_parameters[i][1]/2);
        //cout<<"mu: "<<input_parameters[i][0]<<" sigma: "<<input_parameters[i][1]<<endl;
        for(int j=0; j<M; ++j){
            number=distribution(generator);
            while(number>upper_bound||number<lower_bound){
                number=distribution(generator);
            }
            random_parameters[j][i]=number;
            //cout<<random_parameters[j][i]<<endl;
        }
    }
            
}

int main(int argc, char** argv)
{
    time_t start, end;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    //printf("Hello world from processor %s, rank %d out of %d processors\n",processor_name, world_rank, world_size);

    vector< vector<double> > input_parameters(N_Input, vector<double>(2));
    vector< vector<double> > random_parameters(M, vector<double>(N_Input));
    vector<vector <double> > MP_Input(1,vector<double>(MP_sz));    
    
    //double input_parameters[N_Input][2], random_parameters[M][N_Input];
    double threshold=0.05;
    double parameter_list[N_Input]={0.012, 0.012, 0.000250, 0.005};

    for(int i=0; i<N_Input; ++i){
        input_parameters[i][0]=parameter_list[i];
        input_parameters[i][1]=input_parameters[i][0]*threshold;
    }

    MonteCarlo(input_parameters, random_parameters);
   
    string filename="../mission_profile/mission_profile.csv";
    MP_Input=readCsv(filename,world_rank, world_size);

    size_t sz=72;//MP_Input.size();
        
    unordered_map<size_t, vector<size_t> > hash_list=clusterMerging(MP_Input, sz);
    if(world_rank==0)
    	cout<<"In total: "<<hash_list.size()<<" cases"<<endl;   
    vector<size_t> key_list;

    for(const auto iter : hash_list){
        key_list.push_back(iter.first);
    }

    double *rst = (double*)malloc(N_Output*sizeof(double));
    double *result_all = (double*)malloc(hash_list.size()*N_Output*sizeof(double));
    double *dictionary = (double*)malloc(N_Output*sz*sizeof(double));
    //unordered_map<size_t, vector<double> > dictionary;
    if(world_rank==0){
        time(&start);
    }

    vector<double> vec, tmp;
    std::hash<string> key;
    size_t count=0;
    size_t range=0;
    double *result = (double*)malloc(range*N_Output*sizeof(double));

    if(hash_list.size()<=world_size){
 	if(world_rank<hash_list.size())
	{
	       range = 1;
	}
	else{ 
		range = 0;
	}
    }
    else{
    	range = (hash_list.size()+world_size)/world_size;
    }

    for(int i=world_rank*range; i<(world_rank+1)*range; ++i){
        if(i>=hash_list.size())
            continue;
        else{
            size_t current_key=key_list[i];
            vec=MP_Input[0];
            vec=MP_Input[hash_list[current_key][0]];
            size_t hash_value=getHash(vec);
            tmp.push_back(vec[4]);
            tmp.push_back(vec[7]);
            tmp.push_back(vec[8]);
            //memcpy(&result[i-world_rank*range],runSimulation(tmp,i),N_Output*sizeof(double));
            runSimulation(tmp,i);
            //dictionary[hash_value]=vec;
            //cout<<"Simulation No."<<i<<" Finished"<<endl;
            //MPI_Barrier(MPI_COMM_WORLD);
            count++;            
        }
       
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Gather(&result[0], count*N_Output, MPI_DOUBLE, &result_all[world_rank*range*N_Output], count*N_Output, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /*vector<double> data;

    data.push_back(0.5+0.01*world_rank);
    data.push_back(14+world_rank);
    data.push_back(97+world_rank);
    */

        // Traverse
    //unordered_map<size_t, int>::iterator iter;
    //for(iter=dict.begin();iter!=dict.end();++iter)
    //    cout<<iter->first<<ends<<iter->second<<endl;

    if(world_rank==0){
        time(&end);
        double dif = difftime (end,start);
        printf ("Elasped time is %.5lf seconds. \n", dif );

        //set value for each data point
        for(size_t idx=0; idx<key_list.size(); ++idx){   
            size_t hash_val = key_list[idx];
            memcpy(rst, &result_all[idx*N_Output], N_Output*sizeof(double));
            for(size_t itr; itr<hash_list[hash_val].size(); ++itr){
                memcpy(&dictionary[itr*N_Output],&result[idx*N_Output], N_Output*sizeof(double));
            }       
        }
    }

    free(rst);
    free(result);
    free(result_all);
    free(dictionary);

    MPI_Finalize();

    return 0;
}
