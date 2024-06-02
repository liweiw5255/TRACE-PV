#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Define a struct to hold key-value pairs
struct KeyValuePair {
    std::string key;
    std::string value;
};

int main() {
    // Read the JSON file into a string
    std::ifstream file("../../parameters/pv_panel/panel.json");
    std::string jsonData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Define a vector to store key-value pairs
    std::vector<KeyValuePair> data;

    // Parse the JSON array manually
    size_t startPos = jsonData.find("[");
    size_t endPos = jsonData.find("]");
    if (startPos != std::string::npos && endPos != std::string::npos) {
        std::string jsonArray = jsonData.substr(startPos + 1, endPos - startPos - 1);

        // Parse key-value pairs from the JSON array
        size_t pos = 0;
        while ((pos = jsonArray.find("{", pos)) != std::string::npos) {
            KeyValuePair kvp;

            // Extract key
            size_t keyPos = jsonArray.find("\"key\":", pos);
            if (keyPos != std::string::npos) {
                keyPos += 7;
                size_t endKeyPos = jsonArray.find("\"", keyPos);
                kvp.key = jsonArray.substr(keyPos, endKeyPos - keyPos);
            }

            // Extract value
            size_t valuePos = jsonArray.find("\"value\":", pos);
            if (valuePos != std::string::npos) {
                valuePos += 9;
                size_t endValuePos = jsonArray.find("\"", valuePos);
                kvp.value = jsonArray.substr(valuePos, endValuePos - valuePos);
            }

            // Store key-value pair
            data.push_back(kvp);

            // Move to the next element
            pos = jsonArray.find("}", pos);
        }
    }

    // Print the parsed data
    for (const auto& kvp : data) {
        std::cout << "Key: " << kvp.key << ", Value: " << kvp.value << std::endl;
    }

    return 0;
}
