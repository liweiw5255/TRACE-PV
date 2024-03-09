#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../include/cpp/tracepv_parameters.h"

int main() {
    
    std::unordered_map<std::string, std::string> data = {
        {"Ncell" , "96"},
        {"Nstring" , "29"},
        {"Nmodule" , "19"},
        {"Voc" , "64.2"},
        {"Isc" , "5.96"},
        {"Vmp" , "54.7"}, 
        {"Imp" , "5.58"},
        {"Temp_isc" , "0.061745"},
        {"Temp_soc" , "-0.2727"},
        {"IL" , "5.9657"},
        {"I0" , "6.3076e-12"},
        {"DI_factor" , "0.94489"},
        {"Rsh" , "393.2054"},
        {"Rs" , "0.37428"},
        {"iter" , "1000"},
    };
    

    std::ofstream file("data.json");
    if (!file) {
        std::cerr << "Failed to open file for writing" << std::endl;
        return 1;
    }

    file << "{\n    \"data\": [\n";
    bool first = true;
    for (const auto& pair : data) {
        if (!first) {
            file << ",\n";
        }
        first = false;
        file << "        {\n            \"key\": \"" << pair.first << "\",\n            \"value\": \"" << pair.second << "\"\n        }";
    }
    file << "\n    ]\n}";

    file.close();
    std::cout << "JSON data file created: data.json" << std::endl;

    return 0;
}