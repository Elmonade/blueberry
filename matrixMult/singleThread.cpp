#include "read.h"
#include <chrono>
#include <cstring>
#include <iostream>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 512
#define C1 1024

#define R2 1024
#define C2 512

// Clean memory per value
void mulMat(const int (&mat1)[R1 * C1], const int (&mat2)[R2 * C2],
            int (&result)[R1 * C2]) {
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
void mulMatWithCleanMemory(const int (&mat1)[R1 * C1],
                           const int (&mat2)[R2 * C2], int (&result)[R1 * C2]) {
  memset(result, 0, sizeof(int) * R1 * C2); // Initialize all at once
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] +=
            mat1[i * C1 + k] *
            mat2[k * C2 + j]; // M2 is being read top-down - Transpose this.
      }
    }
  }
}

/*
 * TODO: On tranposed matrix
 */
void mulMatWithCleanMemoryOnTransposed(const int (&mat1)[R1 * C1],
                                       const int (&mat2T)[R2 * C2],
                                       int (&result)[R1 * C2]) {
  memset(result, 0, sizeof(int) * R1 * C2); // Initialize all at once
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
    }
  }
}

void transpose(const int (&ogMat)[R2 * C2], int (&tpMat)[R2 * C2]) {
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
  int mat1[R1 * C1];
  int mat2[R2 * C2];
  int result[R1 * C2];
  int transposed[R2 * C2];

  if (readIntegersFromCSV("matrixMult/2048x1024.csv", mat1, R1 * C1) != R1 * C1) {
    cout << "Error reading first matrix\n";
    return 1;
  }

  if (readIntegersFromCSV("matrixMult/2048x1024.csv", mat2, R2 * C2) != R2 * C2) {
    cout << "Error reading first matrix\n";
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

  return 0;
}
