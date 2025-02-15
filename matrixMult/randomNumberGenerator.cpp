#include <iostream>
#include <fstream>
#include <random>
#include <ctime>

int main() {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
  std::string name = "matrixMult/2048x1024.csv";

    // Define the distribution (e.g., numbers between 1 and 100)
    std::uniform_int_distribution<> dis(1, 100);

    // Open file for writing
    std::ofstream file(name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file!" << std::endl;
        return 1;
    }

  //2048 X 1024
    const int count = 2097152;
    for(int i = 0; i < count; i++) {
        file << dis(gen) << ",";

        // Add progress indicator
        if (i % 6000 == 0) {
            std::cout << "Progress: " << (i * 100 / count) << "%" << std::endl;
        }
    }

    file.close();
    std::cout << "File generated: " << name << std::endl;

    return 0;
}
