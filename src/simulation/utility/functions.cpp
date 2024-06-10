#include "../include/a2s_header.h"
#include "../include/a2s_parameters.h"

void read_csv(const std::string& filename, std::vector<std::vector<double>>& data) {
   std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                double value = std::stod(token);
                row.push_back(value);
            } catch (const std::invalid_argument& e) {
                // Handle invalid tokens if needed
                std::cerr << "Invalid token: " << token << std::endl;
            }
        }
        data.push_back(row);
    }

    file.close();
}

void find_min_max(const double* data, int size, double& minVal, double& maxVal) {
    if (size <= 0) {
        // Handle empty array case
        std::cerr << "Error: Empty array" << std::endl;
        return;
    }

    // Initialize minVal and maxVal with the first element of the array
    minVal = maxVal = data[0];

    // Iterate through the array starting from the second element
    for (int i = 1; i < size; ++i) {
        // Update minVal if the current element is smaller
        if (data[i] < minVal) {
            minVal = data[i];
        }
        // Update maxVal if the current element is larger
        if (data[i] > maxVal) {
            maxVal = data[i];
        }
    }
}

//double getSRated(double T, double Ir, int mode){




 //   return mode==1? 1:0;

//}
     

