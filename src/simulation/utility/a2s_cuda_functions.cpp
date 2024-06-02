#include "../include/a2s_header.h"
#include "../include/a2s_parameters.h"

void read_csv(const std::string& filename, double *&data_array) {
    std::vector<std::vector<double>> data;
    int rows, cols;

    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }

    std::string line;
    double value;
    // Read each line from the file
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        // Read each cell from the line
        while (std::getline(ss, cell, ',')) {
            value = std::stod(cell); 
            row.push_back(value);
        }

        // Add row to data
        data.push_back(row);
    }

    data_array = new double[data.size() * data[0].size()];
    
    for(int i=0; i<data.size(); ++i){
        for(int j=0; j<data[0].size(); ++j){
            data_array[i * data[0].size() + j] = data[i][j];
        }
    }

    // Close the file
    file.close();

}
     

