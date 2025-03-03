#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

int main() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::string A = "data/A.csv";
  std::string B = "data/B.csv";

  std::uniform_real_distribution<double> dis(0, 2048);

  std::ofstream fileA(A);
  if (!fileA.is_open()) {
    std::cerr << "Error: Could not open file!" << std::endl;
    return 1;
  }
  std::ofstream fileB(B);
  if (!fileB.is_open()) {
    std::cerr << "Error: Could not open file!" << std::endl;
    return 1;
  }

  const int rows = 2048;
  const int cols = 2048;

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fileA << dis(gen);
      fileB << dis(gen);
      if (j < cols - 1) {
        fileA << ",";
        fileB << ",";
      }
    }
    if (i < rows - 1) {
      fileA << "\n";
      fileB << "\n";
    }

    // Progress indicator
    if (i % 100 == 0) {
      std::cout << "Progress: " << (i * 100 / rows) << "%" << std::endl;
    }
  }

  fileA.close();
  fileB.close();
  std::cout << "File generated: " << A << std::endl;
  std::cout << "File generated: " << B << std::endl;
  return 0;
}
