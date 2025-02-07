#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "read.h"

int readIntegersFromCSV(const std::string& filename, int arr[], int num_integers) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 0;
    }

    std::string line, cell;

    // Read the first line
    if (std::getline(file, line)) {
        std::istringstream ss(line);

        // Read specified number of integers
        for (int i = 0; i < num_integers; ++i) {
            if (std::getline(ss, cell, ',')) {
                try {
                    arr[i] = std::stoi(cell);
                } catch (...) {
                    std::cerr << "Error: Invalid integer at position " << i << std::endl;
                    return i;
                }
            } else {
                std::cerr << "Error: Not enough integers in the file" << std::endl;
                return i;
            }
        }
    }

    file.close();
    return num_integers;
}
