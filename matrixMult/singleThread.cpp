#include <chrono>
#include <iostream>
#include "read.h"
#include <cstring>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 3
#define C1 2
#define R2 2
#define C2 3

//Clean memory per value
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

/*
* This gives us slightly faster performance
*/
void mulMatWithCleanMemory(const int (&mat1)[R1 * C1], const int (&mat2)[R2 * C2], int (&result)[R1 * C2]) {
    memset(result, 0, sizeof(int) * R1 * C2);  // Initialize all at once
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
void mulMatWithCleanMemoryOnTransposed(const int (&mat1T)[R1 * C1], const int (&mat2)[R2 * C2], int (&result)[R1 * C2]) {
  memset(result, 0, sizeof(int) * R1 * C2);  // Initialize all at once
  for (int i = 0; i < R1; i++) {
      for (int j = 0; j < C1; j++) {
          for (int k = 0; k < C1; k++) {
              result[i * C1 + j] += mat1T[k * C2 + j] * mat2[k * C2 + j];
          }
      }
  }
}

void transpose(const int(&mat1)[R1*C1], int(&mat1T)[R1*C1]) {

  cout << "\n\n Before transposing" << R1 <<"x"<< C1<<"\n";

  for (int i = 0; i < C1*R1; i++) {
    cout << mat1[i] << "\n";
  }

  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C1; j++) {
      cout << C1*i+j<<"-"<< mat1[C1*i + j] << "\t";
    }
    cout << "\n";
  }
  cout << "\n";



  // Row -> Column
  // 1, 2, 3, 4 -> 1, 3, 2, 4
  memset(mat1T, 0, sizeof(int) * R1 * C2);
  int cnt =  0;
  for (int i = 0; i < C1; i++) {
    for (int j = 0; j < R1; j++) {
      int lemon = mat1[C1*j + i];
      cout << C1*j+i << " --- "<<lemon << "\n";
      mat1T[cnt] = lemon;
      cnt++;
    }
  }


  cout << "\n\n After transposing\n";

  for (int i = 0; i < C1*R1; i++) {
    cout << mat1T[i] << "\n";
  }


  for (int i = 0; i < C1; i++) {
    for (int j = 0; j < R1; j++) {
      cout << mat1T[R1*i + j] << "\t";
    }
    cout << "\n";
  }
  cout << "\n";

}

int main() {
    int mat1[R1 * C1];
    int mat2[R2 * C2];
    int result[R1 * C2];
    int mat1T[R1 * C1];

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

    auto t1 = high_resolution_clock::now();
    mulMatWithCleanMemory(mat1, mat2, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "Normal Time = " << ms_double.count() << "ms\n";

    cout << "First few elements of result matrix:\n";
    for (int i = 0; i < 5; i++) {
        cout << result[i] << " ";
    }
    cout << "...\n";



    transpose(mat1, mat1T);
    t1 = high_resolution_clock::now();
    mulMatWithCleanMemoryOnTransposed(mat1T, mat2, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "Transposed Time = " << ms_double.count() << "ms\n";

    cout << "First few elements of result matrix:\n";
    for (int i = 0; i < 5; i++) {
        cout << result[i] << " ";
    }
    cout << "...\n";



    return 0;
}
