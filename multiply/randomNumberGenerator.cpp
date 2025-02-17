#include <iostream>
#include <fstream>
#include <random>
#include <ctime>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::string name = "multiply/2048x2048.csv";
    
    std::uniform_int_distribution<> dis(0, 100);
    
    std::ofstream file(name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file!" << std::endl;
        return 1;
    }

    const int rows = 2048;
    const int cols = 2048;
    
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            file << dis(gen);
            if (j < cols - 1) {
                file << ",";
            }
        }
        if (i < rows - 1) {
            file << "\n";
        }
        
        // Progress indicator
        if (i % 100 == 0) {
            std::cout << "Progress: " << (i * 100 / rows) << "%" << std::endl;
        }
    }

    file.close();
    std::cout << "File generated: " << name << std::endl;
    return 0;
}
