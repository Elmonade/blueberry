#include <iostream>
#include <fstream>
#include <random>
#include <ctime>

int main() {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution (e.g., numbers between 1 and 100)
    std::uniform_int_distribution<> dis(1, 100);

    // Open file for writing
    std::ofstream file("matrixMult/random_numbers.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file!" << std::endl;
        return 1;
    }

    // Generate and write 60000 random numbers
    const int count = 60000;
    for(int i = 0; i < count; i++) {
        file << dis(gen) << ",";

        // Add progress indicator
        if (i % 6000 == 0) {
            std::cout << "Progress: " << (i * 100 / count) << "%" << std::endl;
        }
    }

    file.close();
    std::cout << "Complete! 60000 random numbers have been saved to random_numbers.csv" << std::endl;

    return 0;
}