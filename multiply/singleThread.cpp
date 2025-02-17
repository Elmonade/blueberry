#include "read.h"
#include <chrono>
#include <cstring>
#include <iostream>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 1024
#define C1 2048

#define R2 2048
#define C2 1024

// Clean memory per value
void mulMat(const std::vector<int>& mat1, const std::vector<int>& mat2,
            std::vector<int>& result) {
    result.resize(R1 * C2);
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
void mulMatWithCleanMemory(const std::vector<int>& mat1, 
                          const std::vector<int>& mat2, 
                          std::vector<int>& result) {
    result.resize(R1 * C2);
    std::fill(result.begin(), result.end(), 0);
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
void mulMatWithCleanMemoryOnTransposed(const std::vector<int>& mat1,
                                      const std::vector<int>& mat2T,
                                      std::vector<int>& result) {
    result.resize(R1 * C2, 0);  // Resize and initialize all elements to 0

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
void mulMatWithUnrolled(const std::vector<int>& mat1,
                                      const std::vector<int>& mat2T,
                                      std::vector<int>& result) {
    result.resize(R1 * C2, 0);  // Resize and initialize all elements to 0
  const int stepsize = 2;
  for (int i = 0; i < ROUND_DOWN(R1, stepsize); i += stepsize) {
    for (int j = 0; j < ROUND_DOWN(C2, stepsize); j += stepsize) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];

        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
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
void mulMatWithUnrolledAll(const std::vector<int>& mat1,
                                      const std::vector<int>& mat2T,
                                      std::vector<int>& result) {
    result.resize(R1 * C2, 0);  // Resize and initialize all elements to 0
  const int stepsize = 2;
  for (int i = 0; i < ROUND_DOWN(R1, stepsize); i += stepsize) {
    for (int j = 0; j < ROUND_DOWN(C2, stepsize); j += stepsize) {
      for (int k = 0; k < ROUND_DOWN(C1, stepsize); k += stepsize) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + j] += mat1[i * C1 + k+1] * mat2T[j * C1 + k+1];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k+1] * mat2T[(j + 1) * C1 + k+1];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k+1] * mat2T[j * C1 + k+1];
        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + (k+1)] * mat2T[(j + 1) * C1 + (k+1)];
      }
      // Handle remaining k
      for (int k = ROUND_DOWN(C1, stepsize); k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
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

void mulMatBlocked(const std::vector<int>& mat1,
                                      const std::vector<int>& mat2T,
                                      std::vector<int>& result) {
  const int BLOCK_SIZE = 64; // Cache size
    result.resize(R1 * C2, 0);  // Resize and initialize all elements to 0

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {   // Block row of mat1
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) { // Block column of mat2T
      for (int k0 = 0; k0 < C1;
           k0 += BLOCK_SIZE) { // Block for shared dimension
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

void transpose(const std::vector<int>& ogMat, std::vector<int>& tpMat) {
    tpMat.resize(R2 * C2, 0);
    
    int cnt = 0;
    for (int i = 0; i < C2; i++) {
        for (int j = 0; j < R2; j++) {
            tpMat[cnt] = ogMat[C2 * j + i];
            cnt++;
        }
    }
}

int main() {
  std::vector<int> mat1;
  std::vector<int> mat2;
  std::vector<int> result;
  std::vector<int> transposed;

  if (readIntegersFromCSV("multiply/2048x2048.csv", mat1, R1, C1) != R1 * C1) {
    cout << "Error reading matrix 1\n";
    return 1;
  }

  if (readIntegersFromCSV("multiply/2048x2048.csv", mat2, R2, C2) != R2 * C2) {
    cout << "Error reading matrix 2\n";
    return 1;
  }

  if (C1 != R2) {
    cout << "Matrix 1 columns doesn't match Matrix 2 rows.\n";
    exit(EXIT_FAILURE);
  }

  auto t1 = high_resolution_clock::now();
  mulMatWithCleanMemory(mat1, mat2, result);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout << "Normal Time = " << ms_double.count() << "ms\n";

  for (int i = 0; i < 5; i++) {
    cout << result[i] << " ";
  }
  cout << "\n";

  transpose(mat2, transposed);
  t1 = high_resolution_clock::now();
  mulMatWithCleanMemoryOnTransposed(mat1, transposed, result);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;

  std::cout << "Transposed Time = " << ms_double.count() << "ms\n";

  for (int i = 0; i < 5; i++) {
    cout << result[i] << " ";
  }
  cout << "\n";

  t1 = high_resolution_clock::now();
  mulMatWithUnrolled(mat1, transposed, result);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;

  std::cout << "Unrolled Time = " << ms_double.count() << "ms\n";

  for (int i = 0; i < 5; i++) {
    cout << result[i] << " ";
  }
  cout << "\n";

  t1 = high_resolution_clock::now();
  mulMatWithUnrolledAll(mat1, transposed, result);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;

  std::cout << "Unrolled All Time = " << ms_double.count() << "ms\n";

  for (int i = 0; i < 5; i++) {
    cout << result[i] << " ";
  }
  cout << "\n";

  t1 = high_resolution_clock::now();
  mulMatBlocked(mat1, transposed, result);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;

  std::cout << "Blocked Time = " << ms_double.count() << "ms\n";

  for (int i = 0; i < 5; i++) {
    cout << result[i] << " ";
  }
  cout << "\n";

  return 0;
}
