#include <cblas.h>
#include <chrono>
#include <cstdlib> // ENV
#include <cstring>
#include <iostream>
#include "read.h"
#include <x86intrin.h>
#include <immintrin.h>  // For AVX/FMA intrinsics

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2043
#define C1 2043

#define R2 2043
#define C2 2043

void normal(const double *mat1, const double *mat2, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      for (int k = 0; k < C1; k++) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
      }
    }
  }
}

#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void locality_avx512(const double *mat1, const double *mat2T, double *result) {
  int unroll = 8; // Process 8 doubles at once with AVX-512
  memset(result, 0, sizeof(double) * R1 * C2);
  
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      __m512d sum_vec = _mm512_setzero_pd(); // 512-bit zero vector
      
      // Process 8 elements at a time
      int limit = ROUND_DOWN(C1, unroll);
      for (int k = 0; k < limit; k += unroll) {
        __m512d A_vec = _mm512_loadu_pd(&mat1[i * C1 + k]); // Load 8 doubles
        __m512d B_vec = _mm512_loadu_pd(&mat2T[j * C1 + k]); // Load 8 doubles
        
        // Fused multiply-add
        sum_vec = _mm512_fmadd_pd(A_vec, B_vec, sum_vec);
      }
      
      // Horizontal sum (much simpler with AVX-512)
      double dotProduct = _mm512_reduce_add_pd(sum_vec);
      
      // Handle remaining elements with scalar operations
      for (int k = limit; k < C1; k++) {
        dotProduct += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
      
      result[i * C2 + j] = dotProduct;
    }
  }
}
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void locality(const double *mat1, const double *mat2T, double *result) {
  int UNROLL = 8;
  memset(result, 0, sizeof(double) * R1 * C2);
  
  for (int i = 0; i < R1; i++) {
    for (int j = 0; j < C2; j++) {
      __m256d sum_vec = _mm256_setzero_pd();
      
      int limit = ROUND_DOWN(C1, UNROLL);
      for (int k = 0; k < limit; k += UNROLL) {
        // Load 8x64bit 
        __m256d A_vec_0 = _mm256_loadu_pd(&mat1[i * C1 + k]);
        __m256d A_vec_1 = _mm256_loadu_pd(&mat1[i * C1 + k + 4]);
        
        // Load 8x64bit
        __m256d B_vec_0 = _mm256_loadu_pd(&mat2T[j * C1 + k]);
        __m256d B_vec_1 = _mm256_loadu_pd(&mat2T[j * C1 + k + 4]);
        
        // FUSED MULTIPLY ADD -> A*B+=SUM
        sum_vec = _mm256_fmadd_pd(A_vec_0, B_vec_0, sum_vec);
        sum_vec = _mm256_fmadd_pd(A_vec_1, B_vec_1, sum_vec);
      }
      
      // Horizontal sum of vector to get final result
      __m128d sum_high = _mm256_extractf128_pd(sum_vec, 1);
      __m128d sum_low = _mm256_castpd256_pd128(sum_vec);
      __m128d sum_hl = _mm_add_pd(sum_high, sum_low);
      __m128d sum_lh = _mm_permute_pd(sum_hl, 1);
      __m128d scalar_sum = _mm_add_sd(sum_hl, sum_lh);
      
      double dotProduct = _mm_cvtsd_f64(scalar_sum);
      
      // Handle remaining elements with scalar operations
      for (int k = limit; k < C1; k++) {
        dotProduct += mat1[i * C1 + k] * mat2T[j * C1 + k];
      }
      
      result[i * C2 + j] = dotProduct;
    }
  }
}
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void broadcastedSIMDrollFour(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 4;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i + 1) * C2 + j];
            double sum2 = result[(i + 2) * C2 + j];
            double sum3 = result[(i + 3) * C2 + j];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += 2) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];
              
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
              
              sum2 += mat1[(i + 2) * C1 + k] * mat2T[j * C1 + k];
              sum2 += mat1[(i + 2) * C1 + k + 1] * mat2T[j * C1 + k + 1];
              
              sum3 += mat1[(i + 3) * C1 + k] * mat2T[j * C1 + k];
              sum3 += mat1[(i + 3) * C1 + k + 1] * mat2T[j * C1 + k + 1];
            }
            
            // Handle remaining k elements if C1 is not even
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), 2); k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum2 += mat1[(i + 2) * C1 + k] * mat2T[j * C1 + k];
              sum3 += mat1[(i + 3) * C1 + k] * mat2T[j * C1 + k];
            }
            
            // Store results
            result[i * C2 + j] = sum0;
            result[(i + 1) * C2 + j] = sum1;
            result[(i + 2) * C2 + j] = sum2;
            result[(i + 3) * C2 + j] = sum3;
          }
        }
        
        // Handle remaining rows that couldn't be unrolled
        for (int i = ROUND_DOWN(std::min(i0 + BLOCK_SIZE, R1), UNROLL);
             i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum = result[i * C2 + j];
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
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void broadcastedSIMD(const double *mat1, const double *mat2T, double *result) {

  const int BLOCK_SIZE = 64;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {

        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i + 1) * C2 + j];

            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += 2) {

              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];

              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
            }

            // Handle remaining k elements if C1 is not even
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), 2); k < std::min(k0 + BLOCK_SIZE, C1); k++) {
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
            double sum = result[i * C2 + j];
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

void broadcasted(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 128;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {

        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i + 1) * C2 + j];

            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += 2) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];

              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];
            }

            // Handle remaining k elements if C1 is not even
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), 2); k < std::min(k0 + BLOCK_SIZE, C1);
                 k++) {
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
            double sum = result[i * C2 + j];
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

    if (readDoubleFromCSV("multiply/A.csv", mat1, matrix1_size) != matrix1_size) {
      std::cout << "Error reading matrix A\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    if (readDoubleFromCSV("multiply/B.csv", mat2, matrix2_size) != matrix2_size) {
      std::cout << "Error reading matrix B\n";
      delete[] mat1;
      delete[] mat2;
      delete[] result;
      delete[] transposed;
      return 1;
    }

    transpose(mat2, transposed);

    auto t1 = high_resolution_clock::now();
    normal(mat1, mat2, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nBlocked Time = " << ms_double.count() << "ms\n";
    std::cout << "Normal = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/normal.csv", R1, C2);


    t1 = high_resolution_clock::now();
    locality_avx512(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nSIMD Time = " << ms_double.count() << "ms\n";
    std::cout << "SIMD= " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/simd.csv", R1, C2);


    t1 = high_resolution_clock::now();
    broadcasted(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nBroadcasted Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked and Unrolled Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "multiply/broadcasted.csv", R1, C2);

    // Add OpenBLAS
    t1 = high_resolution_clock::now();
    // OpenBLAS matrix multiplication using DGEMM
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
    writeMatrixToCSV(result, "multiply/openblas.csv", R1, C2);

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
