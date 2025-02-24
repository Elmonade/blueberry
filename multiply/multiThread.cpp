#include <Eigen/Dense>
#include <chrono>
#include <cstring>
#include <iostream>
#include "read.h"
#include <omp.h>

using Eigen::MatrixXd; // Add Eigen matrix type
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 1024
#define C1 2048

#define R2 2048
#define C2 1024

#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void mulMatBlockedWithUnroll(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 128;
  const int UNROLL = 2;
  memset(result, 0, sizeof(int) * R1 * C2);
  // Block
  Eigen::initParallel();
  #pragma omp parallel for
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Unroll
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            int sum0 = result[i * C2 + j];
            int sum1 = result[(i + 1) * C2 + j];
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += 2) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];

              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
            }
            // C1 is odd
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), 2);
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
            }

            result[i * C2 + j] = sum0;
            result[(i + 1) * C2 + j] = sum1;
          }
        }

        // R1 is odd
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

    if (readDoubleFromCSV("multiply/2048x2048.csv", mat1, matrix1_size) !=
        matrix1_size) {
      std::cout << "Error reading matrix 1\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    if (readDoubleFromCSV("multiply/2048x2048.csv", mat2, matrix2_size) !=
        matrix2_size) {
      std::cout << "Error reading matrix 2\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    auto t1 = high_resolution_clock::now();
    mulMatBlockedWithUnroll(mat1, mat2, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nTime = " << ms_double.count() << "ms\n";
    std::cout << "Performance = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/multiThread.csv", R1, C2);

    // Add Eigen measurement
    MatrixXd eigenMat1 = MatrixXd::Map(mat1, R1, C1);
    MatrixXd eigenMat2 = MatrixXd::Map(mat2, R2, C2);
    MatrixXd eigenResult(R1, C2);

    t1 = high_resolution_clock::now();
    eigenResult = eigenMat1 * eigenMat2;
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "Eigen Time = " << ms_double.count() << "ms\n";
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
