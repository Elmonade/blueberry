#include <chrono>
#include <iostream>
#include "read.h"
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 200
#define C1 300
#define R2 300
#define C2 200

void mulMat(const int (&mat1)[R1 * C1], const int (&mat2)[R2 * C2], int (&result)[R1 * C2]) {
    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            result[i * C2 + j] = 0;
            for (int k = 0; k < C1; k++) {
                result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
            }
        }
    }
}

int main() {
    int mat1[R1 * C1];
    int mat2[R2 * C2];
    int result[R1 * C2];

    if (readIntegersFromCSV("matrixMult/random_numbers.csv", mat1, R1 * C1) != R1 * C1) {
        cout << "Error reading first matrix\n";
        return 1;
    }

    if (readIntegersFromCSV("matrixMult/random_numbers.csv", mat2, R2 * C2) != R1 * C1) {
        cout << "Error reading first matrix\n";
        return 1;
    }

    if (C1 != R2) {
        cout << "Matrix 1 columns doesn't match Matrix 2 rows.\n";
        exit(EXIT_FAILURE);
    }

    const auto t1 = high_resolution_clock::now();
    mulMat(mat1, mat2, result);
    const auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "Time = " << ms_double.count() << "ms\n";

    cout << "First few elements of result matrix:\n";
    for (int i = 0; i < 5; i++) {
        cout << result[i] << " ";
    }
    cout << "...\n";

    return 0;
}
