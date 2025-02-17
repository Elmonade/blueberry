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
  const int BLOCK_SIZE = 64; // Cache size
  memset(result, 0, sizeof(int) * R1 * C2);

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

int main() {
  try {
    const size_t matrix1_size = (long long)R1 * C1; // 1024 * 2048
    const size_t matrix2_size = (long long)R2 * C2; // 2048 * 1024
    const size_t result_size = (long long)R1 * C2;  // 1024 * 1024 for result

    // Allocate arrays on heap
    int *mat1 = new int[matrix1_size];
    int *mat2 = new int[matrix2_size];
    int *result = new int[result_size];      // Result is R1 x C2
    int *transposed = new int[matrix2_size]; // Same size as mat2

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
