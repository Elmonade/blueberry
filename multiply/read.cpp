#include "read.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

double readDoubleFromCSV(const std::string &filename, double* arr, int num_doubles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 0;
    }

    std::string line;
    int total_integers = 0;
    const int rows = 2048;
    const int cols = 2048;

    // Read row by row
    for (int i = 0; i < rows && total_integers < num_doubles; i++) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Unexpected end of file at row " << i << std::endl;
            file.close();
            return total_integers;
        }

        std::istringstream ss(line);
        std::string cell;
        
        // Read each number in the current row
        for (int j = 0; j < cols && total_integers < num_doubles; j++) {
            if (std::getline(ss, cell, ',')) {
                try {
                    arr[total_integers] = std::stoi(cell);
                    total_integers++;
                } catch (...) {
                    std::cerr << "Error: Invalid integer at position (" << i << "," << j << ")" << std::endl;
                    file.close();
                    return total_integers;
                }
            } else {
                std::cerr << "Error: Not enough integers in row " << i << std::endl;
                file.close();
                return total_integers;
            }
        }
    }

    file.close();
    return total_integers;
}

void writeMatrixToCSV(const double* matrix, const char* filename, int rows, int cols) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFile << matrix[i * cols + j];
            if (j < cols - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
}
