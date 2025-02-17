#include "read.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int readIntegersFromCSV(const std::string &filename, std::vector<int> &matrix,
                        int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 0;
    }

    // Pre-allocate the vector to avoid reallocation
    matrix.reserve(rows * cols);
    
    std::string line;
    int total_integers = 0;
    
    // Read line by line
    while (std::getline(file, line) && total_integers < rows * cols) {
        std::istringstream ss(line);
        std::string cell;
        
        // Process each number in the current line
        while (std::getline(ss, cell, ',') && total_integers < rows * cols) {
            try {
                matrix.push_back(std::stoi(cell));
                total_integers++;
            } catch (...) {
                std::cerr << "Error: Invalid integer at position " << total_integers << std::endl;
                return total_integers;
            }
        }
    }

    file.close();
    
    if (total_integers != rows * cols) {
        std::cerr << "Error: Expected " << rows * cols << " integers, but got " 
                  << total_integers << std::endl;
    }
    
    return total_integers;
}
