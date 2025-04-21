#include <cblas.h>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include "read.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2048
#define C1 2048

#define R2 2048
#define C2 2048

void multiply(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 256;
  const int UNROLL = 8; // 8 * double = 512

  memset(result, 0, sizeof(double) * R1 * C2);

  #pragma omp parallel for collapse(3)
  //BLOCK
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        //UNROLL
        for (int i = i0; i < i0 + BLOCK_SIZE; i += UNROLL) {
          for (int j = j0; j < j0 + BLOCK_SIZE; j++) {
            // Broadcast 0.0 to whole vector.
            __m512d sum0 = _mm512_set1_pd(0.0);
            __m512d sum1 = _mm512_set1_pd(0.0);
            __m512d sum2 = _mm512_set1_pd(0.0);
            __m512d sum3 = _mm512_set1_pd(0.0);
            __m512d sum4 = _mm512_set1_pd(0.0);
            __m512d sum5 = _mm512_set1_pd(0.0);
            __m512d sum6 = _mm512_set1_pd(0.0);
            __m512d sum7 = _mm512_set1_pd(0.0);

            for (int k = k0; k < k0 + BLOCK_SIZE; k += UNROLL) {
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

    double custom_time = ms_double.count();
    std::cout << "\nCustom Multiplication Time = " << custom_time << "ms\n";
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

    writeTimingToCSV("data/time.csv", custom_time, ms_double.count());

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
