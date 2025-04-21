#include "read.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

double readDoubleFromCSV(const std::string &filename, double *arr, int num_doubles) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return 0;
  }

  std::string line;
  int total_doubles = 0;
  const int rows = 4096;
  const int cols = 4096;

  // Read row by row
  for (int i = 0; i < rows && total_doubles < num_doubles; i++) {
    if (!std::getline(file, line)) {
      std::cerr << "Error: Unexpected end of file at row " << i << std::endl;
      file.close();
      return total_doubles;
    }

    std::istringstream ss(line);
    std::string cell;

    // Read each number in the current row
    for (int j = 0; j < cols && total_doubles < num_doubles; j++) {
      if (std::getline(ss, cell, ',')) {
        try {
          arr[total_doubles] = std::stod(cell);
          total_doubles++;
        } catch (...) {
          std::cerr << "Error: Invalid double at position (" << i << "," << j << ")" << std::endl;
          file.close();
          return total_doubles;
        }
      } else {
        std::cerr << "Error: Not enough integers in row " << i << std::endl;
        file.close();
        return total_doubles;
      }
    }
  }

  file.close();
  return total_doubles;
}

void writeMatrixToCSV(const double* matrix, const char* filename, int rows, int cols) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }
    
    // Match NumPy's standard scientific notation format
    outFile << std::scientific << std::setprecision(5);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Apply truncation instead of rounding to match NumPy behavior
            double value = matrix[i * cols + j];
            char buffer[50];
            std::sprintf(buffer, "%.5e", value); // Format with exactly 5 digits
            outFile << buffer;
            
            if (j < cols - 1) outFile << ",";
        }
        outFile << "\n";
    }
    outFile.close();
}

void writePlotDataToCSV(const std::vector<PlotData>& plotData, const char* filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }
    
    outFile << "Algorithm,Time_ms" << std::endl;
    
    for (const auto& data : plotData) {
        outFile << data.label << "," << data.time << std::endl;
    }
    
    outFile.close();
}

// Add this function to write timing info to CSV (appends data)
void writeTimingToCSV(const std::string& filename, double timeA, double timeB) {
  bool fileExists = false;
  std::ifstream checkFile(filename);
  fileExists = checkFile.good();
  
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error opening " << filename << " for writing\n";
    return;
  }

  if (!fileExists)
    file << "Custom,OpenBLAS\n";

  file << timeA << "," << timeB << "\n";
  
  file.close();
}
