#include <cblas.h>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include "read.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define USE_AVX 1
#define USE_FMA 1

#define R1 1024
#define C1 1024
#define R2 1024
#define C2 1024

void multiply(double *a, double *b_trans, double *c, int m, int n, int k) {
  // m = rows of A and C
  // k = cols of A, rows of B (original B)
  // n = cols of B and C
  const int BLOCK_SIZE = 64;

  // Zero out result matrix
  memset(c, 0, sizeof(double) * m * n);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < m; i += BLOCK_SIZE) {
    for (int j = 0; j < n; j += BLOCK_SIZE) {
      for (int l = 0; l < k; l += BLOCK_SIZE) {
        int i_end = std::min(i + BLOCK_SIZE, m);
        int j_end = std::min(j + BLOCK_SIZE, n);
        int l_end = std::min(l + BLOCK_SIZE, k);

        for (int ii = i; ii < i_end; ii++) {
          for (int jj = j; jj < j_end; jj += 8) { // AVX-512 processes 8 doubles
            __m512d c_vec = _mm512_loadu_pd(c + ii * n + jj);

            for (int ll = l; ll < l_end; ll += 4) { // Unroll by 4
              if (ll + 3 < l_end) {
                __m512d a_vec0 = _mm512_set1_pd(a[ii * k + ll]);
                __m512d a_vec1 = _mm512_set1_pd(a[ii * k + ll + 1]);
                __m512d a_vec2 = _mm512_set1_pd(a[ii * k + ll + 2]);
                __m512d a_vec3 = _mm512_set1_pd(a[ii * k + ll + 3]);

                __m512d b_vec0 = _mm512_loadu_pd(b_trans + ll * n + jj);
                __m512d b_vec1 = _mm512_loadu_pd(b_trans + (ll + 1) * n + jj);
                __m512d b_vec2 = _mm512_loadu_pd(b_trans + (ll + 2) * n + jj);
                __m512d b_vec3 = _mm512_loadu_pd(b_trans + (ll + 3) * n + jj);

                c_vec = _mm512_fmadd_pd(a_vec0, b_vec0, c_vec);
                c_vec = _mm512_fmadd_pd(a_vec1, b_vec1, c_vec);
                c_vec = _mm512_fmadd_pd(a_vec2, b_vec2, c_vec);
                c_vec = _mm512_fmadd_pd(a_vec3, b_vec3, c_vec);
              } else {
                for (; ll < l_end; ll++) {
                  __m512d a_vec = _mm512_set1_pd(a[ii * k + ll]);
                  __m512d b_vec = _mm512_loadu_pd(b_trans + ll * n + jj);
                  c_vec = _mm512_fmadd_pd(a_vec, b_vec, c_vec);
                }
              }
            }
            _mm512_storeu_pd(c + ii * n + jj, c_vec);
          }
        }
      }
    }
  }
}

void transpose(double *ogMat, double *tpMat) {
  memset(tpMat, 0, sizeof(double) * R2 * C2);
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

    // Transpose B after reading
    transpose(mat2, transposed);

    // Custom multiplication
    auto t1 = high_resolution_clock::now();
    multiply(mat1, transposed, result, R1, C2, C1);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nCustom Multiplication Time = " << ms_double.count() << "ms\n";
    std::cout << "Custom Multiplication Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/multiThread.csv", R1, C2);

    // OpenBLAS multiplication
    t1 = high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, // Matrix layout (row major)
                CblasNoTrans, // Don't transpose first matrix
                CblasNoTrans, // Don't transpose second matrix
                R1, // Rows of first matrix
                C2, // Columns of second matrix
                C1, // Columns of first/rows of second matrix
                1.0, // Alpha scaling factor
                mat1, // First matrix
                C1, // Leading dimension of first matrix
                mat2, // Second matrix
                C2, // Leading dimension of second matrix
                0.0, // Beta scaling factor
                result, // Result matrix
                C2); // Leading dimension of result matrix
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nOpenBLAS Time = " << ms_double.count() << "ms\n";
    std::cout << "OpenBLAS Performance = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/openblasMult.csv", R1, C2);

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
