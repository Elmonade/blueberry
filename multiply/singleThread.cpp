#include "read.h"
#include <Eigen/Dense>
#include <chrono>
#include <cstring>
#include <iostream>

using Eigen::MatrixXi; // Add Eigen matrix type
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 1024
#define C1 2048

#define R2 2048
#define C2 1024

// Clean memory per value
void mulMat(const int *mat1, const int *mat2, int *result) {
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      result[i * C2 + j] = 0;
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
      }
    }
  }
}

/*
 * This gives us slightly faster performance
 */
void mulMatWithCleanMemory(const int *mat1, const int *mat2, int *result) {
  memset(result, 0, sizeof(int) * R1 * C2);
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
      }
    }
  }
}

/*
 * TODO: On tranposed matrix
 */
void mulMatWithCleanMemoryOnTransposed(const int *mat1, const int *mat2T,
                                       int *result) {
  memset(result, 0, sizeof(int) * R1 * C2); // Initialize all at once
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
}

/*
 * TODO: Partially unroll(2 of them) the loop
 */
#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
void mulMatWithUnrolled(const int *mat1, const int *mat2T, int *result) {
  memset(result, 0, sizeof(int) * R1 * C2);
  const int stepsize = 2;
  for (int i = 0; i < ROUND_DOWN(R1, stepsize); i += stepsize) {
    for (int j = 0; j < ROUND_DOWN(C2, stepsize); j += stepsize) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];

        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + (j + 1)] +=
            mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
      }
    }
    // Handle remaining columns for these two rows
    for (int j = ROUND_DOWN(C2, stepsize); j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
  // Handle remaining rows
  for (int i = ROUND_DOWN(R1, stepsize); i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
}
/*
 * TODO: Partially unroll(3 of them) the loop
 */
#define ROUND_DOWN(x, s) ((x) & ~((s)-1))
void mulMatWithUnrolledAll(const int *mat1, const int *mat2T, int *result) {
  memset(result, 0, sizeof(int) * R1 * C2);
  const int stepsize = 2;
  for (int i = 0; i < ROUND_DOWN(R1, stepsize); i += stepsize) {
    for (int j = 0; j < ROUND_DOWN(C2, stepsize); j += stepsize) {
      for (int k = 0; k < ROUND_DOWN(C1, stepsize); k += stepsize) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + j] += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[i * C2 + (j + 1)] +=
            mat1[i * C1 + k + 1] * mat2T[(j + 1) * C1 + k + 1];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + j] +=
            mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
        result[(i + 1) * C2 + (j + 1)] +=
            mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[(i + 1) * C2 + (j + 1)] +=
            mat1[(i + 1) * C1 + (k + 1)] * mat2T[(j + 1) * C1 + (k + 1)];
      }
      // Handle remaining k
      for (int k = ROUND_DOWN(C1, stepsize); k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + (j + 1)] +=
            mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
      }
    }
    // Handle remaining columns for these two rows
    for (int j = ROUND_DOWN(C2, stepsize); j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
  // Handle remaining rows
  for (int i = ROUND_DOWN(R1, stepsize); i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
}

void mulMatBlocked(const int *mat1, const int *mat2T, int *result) {
  const int BLOCK_SIZE = 128; // Cache size
  memset(result, 0, sizeof(int) * R1 * C2);

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {

        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            int sum = result[i * C2 + j];
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
            }
            result[i * C2 + j] = sum;
          }
        }
      }
    }
  }
}
/*
 * TODO: Add unrolling to the blocking
 */
void mulMatBlockedWithUnroll(const int *mat1, const int *mat2T, int *result) {
  const int BLOCK_SIZE = 128; // Cache size
  const int UNROLL = 2;       // Unrolling factor
  memset(result, 0, sizeof(int) * R1 * C2);

  // Block level loops
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Within each block
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1);
             i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            int sum0 = result[i * C2 + j];
            int sum1 = result[(i + 1) * C2 + j];

            // Unrolled k-loop for better instruction-level parallelism
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += 2) {
              // First row calculations
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];

              // Second row calculations
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
            }

            // Handle remaining k elements if C1 is not even
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), 2);
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
            }

            // Store results
            result[i * C2 + j] = sum0;
            result[(i + 1) * C2 + j] = sum1;
          }
        }

        // Handle remaining rows that couldn't be unrolled
        for (int i = ROUND_DOWN(std::min(i0 + BLOCK_SIZE, R1), UNROLL);
             i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            int sum = result[i * C2 + j];
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
            }
            result[i * C2 + j] = sum;
          }
        }
      }
    }
  }
}

void transpose(int *ogMat, int *tpMat) {
  memset(tpMat, 0, sizeof(int) * R2 * C2);
  int cnt = 0;
  for (int i = 0; i < C2; i++) {
    for (int j = 0; j < R2; j++) {
      tpMat[cnt] = ogMat[C2 * j + i];
      cnt++;
    }
  }
}

double calculateGFLOPS(double milliseconds) {
  double seconds = milliseconds / 1000.0;
  double operations = 2.0 * R1 * C2 * C1; // 2 operations per multiply-add
  return (operations / seconds) / 1e9;    // Convert to GFLOPS
}

int main() {
  try {
    const size_t matrix1_size = (long long)R1 * C1;
    const size_t matrix2_size = (long long)R2 * C2;
    const size_t result_size = (long long)R1 * C2;

    // Allocate arrays on heap
    int *mat1 = new int[matrix1_size];
    int *mat2 = new int[matrix2_size];
    int *result = new int[result_size];
    int *transposed = new int[matrix2_size];

    if (readIntegersFromCSV("multiply/2048x2048.csv", mat1, matrix1_size) !=
        matrix1_size) {
      std::cout << "Error reading matrix 1\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    if (readIntegersFromCSV("multiply/2048x2048.csv", mat2, matrix2_size) !=
        matrix2_size) {
      std::cout << "Error reading matrix 2\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    auto t1 = high_resolution_clock::now();
    mulMatWithCleanMemory(mat1, mat2, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nNormal Time = " << ms_double.count() << "ms\n";
    std::cout << "Normal Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/normal.csv", R1, C2);

    transpose(mat2, transposed);
    t1 = high_resolution_clock::now();
    mulMatWithCleanMemoryOnTransposed(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nTransposed Time = " << ms_double.count() << "ms\n";
    std::cout << "Transposed Performance = "
              << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/transposed.csv", R1, C2);

    t1 = high_resolution_clock::now();
    mulMatWithUnrolled(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nUnrolled Time = " << ms_double.count() << "ms\n";
    std::cout << "Unrolled Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/unrolled.csv", R1, C2);

    t1 = high_resolution_clock::now();
    mulMatWithUnrolledAll(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nUnrolled All Time = " << ms_double.count() << "ms\n";
    std::cout << "Unrolled All Performance = "
              << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/unrolledAll.csv", R1, C2);

    t1 = high_resolution_clock::now();
    mulMatBlocked(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nBlocked Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/blocked.csv", R1, C2);

    t1 = high_resolution_clock::now();
    mulMatBlockedWithUnroll(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nBlocked and Unrolled Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked and Unrolled Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/blockedUnrolled.csv", R1, C2);

    // Add Eigen measurement
    MatrixXi eigenMat1 = MatrixXi::Map(mat1, R1, C1);
    MatrixXi eigenMat2 = MatrixXi::Map(mat2, R2, C2);
    MatrixXi eigenResult(R1, C2);

    t1 = high_resolution_clock::now();
    eigenResult = eigenMat1 * eigenMat2;
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nEigen Time = " << ms_double.count() << "ms\n";
    std::cout << "Eigen Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/eigen.csv", R1, C2);

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
