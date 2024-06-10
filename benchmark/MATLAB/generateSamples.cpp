#include "simulation.h"

extern void MonteCarlo(vector<vector<double> > input_parameters,vector<vector<double> > random_parameters){
    
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
