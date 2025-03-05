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

#define R1 2048
#define C1 2048
#define R2 2048
#define C2 2048

#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))

void multiply(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 128; //In case we start using something even larger than 2048*2048
  const int UNROLL = 8; // 8 * double = 512

  memset(result, 0, sizeof(double) * R1 * C2);

  #pragma omp parallel for collapse(3)
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // UNROLL
        int iLimit = std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1);
        int jLimit = std::min(j0 + BLOCK_SIZE, C2);
        int kLimit = std::min(k0 + BLOCK_SIZE, C1 - UNROLL + 1);

        for (int i = i0; i < iLimit; i += UNROLL) {
          for (int j = j0; j < jLimit; j++) {
            // Load current values from local buffer
            __m512d sum0 = _mm512_set1_pd(0.0);
            __m512d sum1 = _mm512_set1_pd(0.0);
            __m512d sum2 = _mm512_set1_pd(0.0);
            __m512d sum3 = _mm512_set1_pd(0.0);
            __m512d sum4 = _mm512_set1_pd(0.0);
            __m512d sum5 = _mm512_set1_pd(0.0);
            __m512d sum6 = _mm512_set1_pd(0.0);
            __m512d sum7 = _mm512_set1_pd(0.0);

            for (int k = k0; k < kLimit; k += UNROLL) {
              // Add prefetching for upcoming data
              //_mm_prefetch((char*)&mat2T[j * C1 + k + 64], _MM_HINT_T0);
              //_mm_prefetch((char*)&mat1[i * C1 + k + 64], _MM_HINT_T0);
              
              __m512d m2 = _mm512_loadu_pd(&mat2T[j * C1 + k]);

              __m512d m1_0 = _mm512_loadu_pd(&mat1[i * C1 + k]);
              sum0 = _mm512_fmadd_pd(m1_0, m2, sum0);

              __m512d m1_1 = _mm512_loadu_pd(&mat1[(i + 1) * C1 + k]);
              sum1 = _mm512_fmadd_pd(m1_1, m2, sum1);

              __m512d m1_2 = _mm512_loadu_pd(&mat1[(i + 2) * C1 + k]);
              sum2 = _mm512_fmadd_pd(m1_2, m2, sum2);

              __m512d m1_3 = _mm512_loadu_pd(&mat1[(i + 3) * C1 + k]);
              sum3 = _mm512_fmadd_pd(m1_3, m2, sum3);

              __m512d m1_4 = _mm512_loadu_pd(&mat1[(i + 4) * C1 + k]);
              sum4 = _mm512_fmadd_pd(m1_4, m2, sum4);

              __m512d m1_5 = _mm512_loadu_pd(&mat1[(i + 5) * C1 + k]);
              sum5 = _mm512_fmadd_pd(m1_5, m2, sum5);

              __m512d m1_6 = _mm512_loadu_pd(&mat1[(i + 6) * C1 + k]);
              sum6 = _mm512_fmadd_pd(m1_6, m2, sum6);

              __m512d m1_7 = _mm512_loadu_pd(&mat1[(i + 7) * C1 + k]);
              sum7 = _mm512_fmadd_pd(m1_7, m2, sum7);
            }

            // Vector -> Scalar
            double hsum0 = _mm512_reduce_add_pd(sum0);
            double hsum1 = _mm512_reduce_add_pd(sum1);
            double hsum2 = _mm512_reduce_add_pd(sum2);
            double hsum3 = _mm512_reduce_add_pd(sum3);
            double hsum4 = _mm512_reduce_add_pd(sum4);
            double hsum5 = _mm512_reduce_add_pd(sum5);
            double hsum6 = _mm512_reduce_add_pd(sum6);
            double hsum7 = _mm512_reduce_add_pd(sum7);

            // Handle remaining k values that couldn't be vectorized
            int kRemainderLimit = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL);
            for (int k = kRemainderLimit; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              double m2_val = mat2T[j * C1 + k];

              hsum0 += mat1[i * C1 + k] * m2_val;
              hsum1 += mat1[(i + 1) * C1 + k] * m2_val;
              hsum2 += mat1[(i + 2) * C1 + k] * m2_val;
              hsum3 += mat1[(i + 3) * C1 + k] * m2_val;
              hsum4 += mat1[(i + 4) * C1 + k] * m2_val;
              hsum5 += mat1[(i + 5) * C1 + k] * m2_val;
              hsum6 += mat1[(i + 6) * C1 + k] * m2_val;
              hsum7 += mat1[(i + 7) * C1 + k] * m2_val;
            }

            // Store to local buffer
            result[i * C2 + j] += hsum0;
            result[(i + 1) * C2 + j] += hsum1;
            result[(i + 2) * C2 + j] += hsum2;
            result[(i + 3) * C2 + j] += hsum3;
            result[(i + 4) * C2 + j] += hsum4;
            result[(i + 5) * C2 + j] += hsum5;
            result[(i + 6) * C2 + j] += hsum6;
            result[(i + 7) * C2 + j] += hsum7;
          }
        }

        // Handle remaining rows that couldn't be unrolled
        for (int i = ROUND_DOWN(std::min(i0 + BLOCK_SIZE, R1), UNROLL); i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Use scalar operations for remaining rows
            double sum = result[i * C2 + j];

            // Try to vectorize k loops even for remaining rows
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - UNROLL + 1); k += UNROLL) {
              __m512d m1 = _mm512_loadu_pd(&mat1[i * C1 + k]);
              __m512d m2 = _mm512_loadu_pd(&mat2T[j * C1 + k]);
              __m512d prod = _mm512_mul_pd(m1, m2);
              sum += _mm512_reduce_add_pd(prod);
            }

            // Handle remaining k values
            int k_limit = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL);
            for (int k = k_limit; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
            }

            result[i * C2 + j] = sum;
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
    multiply(mat1, transposed, result);
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
