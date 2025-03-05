#include <cblas.h>
#include <chrono>
#include <cstdlib> // ENV
#include <cstring>
#include <iostream>
#include "read.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2048
#define C1 2048

#define R2 2048
#define C2 2048

void mulMatWithCleanMemory(const double *mat1, const double *mat2, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
      }
    }
  }
}

double calculateGFLOPS(double milliseconds) {
  double seconds = milliseconds / 1000.0;
  double operations = 2.0 * R1 * C2 * C1; // 2 operations per multiply-add
  return (operations / seconds) / 1e9; // Convert to GFLOPS
}

int main() {
  try {
    const size_t matrix1_size = (long long) R1 * C1;
    const size_t matrix2_size = (long long) R2 * C2;
    const size_t result_size = (long long) R1 * C2;

    // Allocate arrays on heap
    double *mat1 = new double[matrix1_size];
    double *mat2 = new double[matrix2_size];
    double *result = new double[result_size];
    double *transposed = new double[matrix2_size];

    if (readDoubleFromCSV("data/A.csv", mat1, matrix1_size) != matrix1_size) {
      std::cout << "Error reading matrix A\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    if (readDoubleFromCSV("data/B.csv", mat2, matrix2_size) != matrix2_size) {
      std::cout << "Error reading matrix B\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    auto t1 = high_resolution_clock::now();
    mulMatBlocked(mat1, transposed, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nBlocked Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked Performance = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/taskOne.csv", R1, C2);

    // Clean up
    delete[] mat1;
    delete[] mat2;
    delete[] result;
    delete[] transposed;

    return 0;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory allocation failed: " << e.what() << "\n";
    return 1;
  }
}
