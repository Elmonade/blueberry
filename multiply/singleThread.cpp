#include <cblas.h>
#include <chrono>
#include <cstdlib> // ENV
#include <cstring>
#include <iostream>
#include "read.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2048
#define C1 2048

#define R2 2048
#define C2 2048

void mulMatWithCleanMemory(const double *mat1, const double *mat2, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2);
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
void mulMatWithCleanMemoryOnTransposed(const double *mat1, const double *mat2T, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2); // Initialize all at once
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
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void mulMatWithUnrolled(const double *mat1, const double *mat2T, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2);
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
#define ROUND_DOWN(x, s) ((x) & ~((s) - 1))
void mulMatWithUnrolledAll(const double *mat1, const double *mat2T, double *result) {
  memset(result, 0, sizeof(double) * R1 * C2);
  const int stepsize = 2;
  for (int i = 0; i < ROUND_DOWN(R1, stepsize); i += stepsize) {
    for (int j = 0; j < ROUND_DOWN(C2, stepsize); j += stepsize) {
      for (int k = 0; k < ROUND_DOWN(C1, stepsize); k += stepsize) {
        result[i * C2 + j] += mat1[i * C1 + k] * mat2T[j * C1 + k];
        result[i * C2 + j] += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];

        result[i * C2 + (j + 1)] += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[i * C2 + (j + 1)] += mat1[i * C1 + k + 1] * mat2T[(j + 1) * C1 + k + 1];

        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
        result[(i + 1) * C2 + j] += mat1[(i + 1) * C1 + k + 1] * mat2T[j * C1 + k + 1];

        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
        result[(i + 1) * C2 + (j + 1)] += mat1[(i + 1) * C1 + (k + 1)] * mat2T[(j + 1) * C1 + (k + 1)];
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

void mulMatBlocked(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64; // Cache size
  memset(result, 0, sizeof(double) * R1 * C2);

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {

        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1); i++) {
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
void mulMatWithUnrolledBlockedI(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 8;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Unroll i by 2
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i + 1) * C2 + j];
            double sum2 = result[(i + 2) * C2 + j];
            double sum3 = result[(i + 3) * C2 + j];
            double sum4 = result[(i + 4) * C2 + j];
            double sum5 = result[(i + 5) * C2 + j];
            double sum6 = result[(i + 6) * C2 + j];
            double sum7 = result[(i + 7) * C2 + j];
            
            // Regular k loop (no unrolling)
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              sum2 += mat1[(i + 2) * C1 + k] * mat2T[j * C1 + k];
              sum3 += mat1[(i + 3) * C1 + k] * mat2T[j * C1 + k];
              sum4 += mat1[(i + 4) * C1 + k] * mat2T[j * C1 + k];
              sum5 += mat1[(i + 5) * C1 + k] * mat2T[j * C1 + k];
              sum6 += mat1[(i + 6) * C1 + k] * mat2T[j * C1 + k];
              sum7 += mat1[(i + 7) * C1 + k] * mat2T[j * C1 + k];
            }
            
            // Store results
            result[i * C2 + j] = sum0;
            result[(i + 1) * C2 + j] = sum1;
            result[(i + 2) * C2 + j] = sum2;
            result[(i + 3) * C2 + j] = sum3;
            result[(i + 4) * C2 + j] = sum4;
            result[(i + 5) * C2 + j] = sum5;
            result[(i + 6) * C2 + j] = sum6;
            result[(i + 7) * C2 + j] = sum7;
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
void mulMatWithUnrolledBlockedK(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 8;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Regular loop for i dimension (no unrolling)
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum = result[i * C2 + j];
            
            // Unrolled k loop
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum += mat1[i * C1 + k + 1] * mat2T[j * C1 + k + 1];
              sum += mat1[i * C1 + k + 2] * mat2T[j * C1 + k + 2];
              sum += mat1[i * C1 + k + 3] * mat2T[j * C1 + k + 3];
              sum += mat1[i * C1 + k + 4] * mat2T[j * C1 + k + 4];
              sum += mat1[i * C1 + k + 5] * mat2T[j * C1 + k + 5];
              sum += mat1[i * C1 + k + 6] * mat2T[j * C1 + k + 6];
              sum += mat1[i * C1 + k + 7] * mat2T[j * C1 + k + 7];
            }
            
            // Handle remaining k elements if C1 is not divisible by UNROLL
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
            }
            
            // Store result
            result[i * C2 + j] = sum;
          }
        }
      }
    }
  }
}

void mulMatWithUnrolledBlockedAll(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Unroll i by 2
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          // Unroll j by 2
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2 - UNROLL + 1); j += UNROLL) {
            // Initialize sums for 2x2 block of results
            double sum00 = result[i * C2 + j];
            double sum01 = result[i * C2 + (j+1)];
            double sum10 = result[(i+1) * C2 + j];
            double sum11 = result[(i+1) * C2 + (j+1)];
            
            // Unroll k by 2
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              // First row, first column
              sum00 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum00 += mat1[i * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
              
              // First row, second column
              sum01 += mat1[i * C1 + k] * mat2T[(j+1) * C1 + k];
              sum01 += mat1[i * C1 + (k+1)] * mat2T[(j+1) * C1 + (k+1)];
              
              // Second row, first column
              sum10 += mat1[(i+1) * C1 + k] * mat2T[j * C1 + k];
              sum10 += mat1[(i+1) * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
              
              // Second row, second column
              sum11 += mat1[(i+1) * C1 + k] * mat2T[(j+1) * C1 + k];
              sum11 += mat1[(i+1) * C1 + (k+1)] * mat2T[(j+1) * C1 + (k+1)];
            }
            
            // Handle remaining k elements if C1 is not divisible by 2
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum00 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum01 += mat1[i * C1 + k] * mat2T[(j+1) * C1 + k];
              sum10 += mat1[(i+1) * C1 + k] * mat2T[j * C1 + k];
              sum11 += mat1[(i+1) * C1 + k] * mat2T[(j+1) * C1 + k];
            }
            
            // Store results
            result[i * C2 + j] = sum00;
            result[i * C2 + (j+1)] = sum01;
            result[(i+1) * C2 + j] = sum10;
            result[(i+1) * C2 + (j+1)] = sum11;
          }
          
          // Handle remaining j elements
          for (int j = ROUND_DOWN(std::min(j0 + BLOCK_SIZE, C2), UNROLL); 
               j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i+1) * C2 + j];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
              sum1 += mat1[(i+1) * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i+1) * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
            }
            
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i+1) * C1 + k] * mat2T[j * C1 + k];
            }
            
            result[i * C2 + j] = sum0;
            result[(i+1) * C2 + j] = sum1;
          }
        }
        
        // Handle remaining i elements
        for (int i = ROUND_DOWN(std::min(i0 + BLOCK_SIZE, R1), UNROLL); 
             i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2 - UNROLL + 1); j += UNROLL) {
            double sum0 = result[i * C2 + j];
            double sum1 = result[i * C2 + (j+1)];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
              sum1 += mat1[i * C1 + k] * mat2T[(j+1) * C1 + k];
              sum1 += mat1[i * C1 + (k+1)] * mat2T[(j+1) * C1 + (k+1)];
            }
            
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[i * C1 + k] * mat2T[(j+1) * C1 + k];
            }
            
            result[i * C2 + j] = sum0;
            result[i * C2 + (j+1)] = sum1;
          }
          
          // Handle remaining j elements for the remaining i
          for (int j = ROUND_DOWN(std::min(j0 + BLOCK_SIZE, C2), UNROLL); 
               j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum = result[i * C2 + j];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum += mat1[i * C1 + (k+1)] * mat2T[j * C1 + (k+1)];
            }
            
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
            }
            
            result[i * C2 + j] = sum;
          }
        }
      }
    }
  }
}
void mulMatWithUnrolledBlockedIJ(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Unroll i by 2
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          // Unroll j by 2
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2 - UNROLL + 1); j += UNROLL) {
            // Initialize sums for 2x2 block of results
            double sum00 = result[i * C2 + j];
            double sum01 = result[i * C2 + (j + 1)];
            double sum10 = result[(i + 1) * C2 + j];
            double sum11 = result[(i + 1) * C2 + (j + 1)];
            
            // Regular k loop (no unrolling)
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              // First row, first column
              sum00 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              
              // First row, second column
              sum01 += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
              
              // Second row, first column
              sum10 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
              
              // Second row, second column
              sum11 += mat1[(i + 1) * C1 + k] * mat2T[(j + 1) * C1 + k];
            }
            
            // Store results
            result[i * C2 + j] = sum00;
            result[i * C2 + (j + 1)] = sum01;
            result[(i + 1) * C2 + j] = sum10;
            result[(i + 1) * C2 + (j + 1)] = sum11;
          }
          
          // Handle remaining j elements for unrolled i
          for (int j = ROUND_DOWN(std::min(j0 + BLOCK_SIZE, C2), UNROLL); 
               j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum0 = result[i * C2 + j];
            double sum1 = result[(i + 1) * C2 + j];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[(i + 1) * C1 + k] * mat2T[j * C1 + k];
            }
            
            result[i * C2 + j] = sum0;
            result[(i + 1) * C2 + j] = sum1;
          }
        }
        
        // Handle remaining i elements
        for (int i = ROUND_DOWN(std::min(i0 + BLOCK_SIZE, R1), UNROLL);
             i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          // Process unrolled j for remaining i
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2 - UNROLL + 1); j += UNROLL) {
            double sum0 = result[i * C2 + j];
            double sum1 = result[i * C2 + (j + 1)];
            
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
            }
            
            result[i * C2 + j] = sum0;
            result[i * C2 + (j + 1)] = sum1;
          }
          
          // Handle remaining j elements for remaining i (corner case)
          for (int j = ROUND_DOWN(std::min(j0 + BLOCK_SIZE, C2), UNROLL); 
               j < std::min(j0 + BLOCK_SIZE, C2); j++) {
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
void mulMatWithUnrolledBlockedJK(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 64;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);
  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {
        // Regular loop for i dimension (no unrolling)
        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1); i++) {
          // Unroll j by 2
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2 - UNROLL + 1); j += UNROLL) {
            // Initialize sums for unrolled columns
            double sum0 = result[i * C2 + j];
            double sum1 = result[i * C2 + (j + 1)];
            
            // Unroll k by 2
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              // For the first column
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum0 += mat1[i * C1 + (k + 1)] * mat2T[j * C1 + (k + 1)];
              
              // For the second column
              sum1 += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
              sum1 += mat1[i * C1 + (k + 1)] * mat2T[(j + 1) * C1 + (k + 1)];
            }
            
            // Handle remaining k elements if C1 is not even
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
              sum0 += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum1 += mat1[i * C1 + k] * mat2T[(j + 1) * C1 + k];
            }
            
            // Store results
            result[i * C2 + j] = sum0;
            result[i * C2 + (j + 1)] = sum1;
          }
          
          // Handle remaining j elements
          for (int j = ROUND_DOWN(std::min(j0 + BLOCK_SIZE, C2), UNROLL);
               j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            double sum = result[i * C2 + j];
            
            // Still unroll k for remaining j
            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
              sum += mat1[i * C1 + k] * mat2T[j * C1 + k];
              sum += mat1[i * C1 + (k + 1)] * mat2T[j * C1 + (k + 1)];
            }
            
            // Handle remaining k elements
            for (int k = ROUND_DOWN(std::min(k0 + BLOCK_SIZE, C1), UNROLL); 
                 k < std::min(k0 + BLOCK_SIZE, C1); k++) {
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
void mulMatWithUnrolledBlocked(const double *mat1, const double *mat2T, double *result) {
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

            for (int k = k0; k < std::min(k0 + BLOCK_SIZE, C1 - 1); k += UNROLL) {
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

void mulMatBlockedWithUnrollByTwo(const double *mat1, const double *mat2T, double *result) {
  const int BLOCK_SIZE = 32;
  const int UNROLL = 2;
  memset(result, 0, sizeof(double) * R1 * C2);

  for (int i0 = 0; i0 < R1; i0 += BLOCK_SIZE) {
    for (int j0 = 0; j0 < C2; j0 += BLOCK_SIZE) {
      for (int k0 = 0; k0 < C1; k0 += BLOCK_SIZE) {

        for (int i = i0; i < std::min(i0 + BLOCK_SIZE, R1 - UNROLL + 1); i += UNROLL) {
          for (int j = j0; j < std::min(j0 + BLOCK_SIZE, C2); j++) {
            // Initialize sums for unrolled rows
            double sum0 = 0;
            double sum1 = 0;

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

    transpose(mat2, transposed);

    auto t1 = high_resolution_clock::now();
    mulMatBlocked(mat1, transposed, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "\nBlocked Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked Performance = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/blocked.csv", R1, C2);


    t1 = high_resolution_clock::now();
    mulMatWithUnrolledAll(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nUnrolled All Time = " << ms_double.count() << "ms\n";
    std::cout << "Unrolled All Performance = " << calculateGFLOPS(ms_double.count()) << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/unrolled.csv", R1, C2);


    t1 = high_resolution_clock::now();
    mulMatWithUnrolledBlockedI(mat1, transposed, result);
    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;

    std::cout << "\nBlocked and Unrolled Time = " << ms_double.count() << "ms\n";
    std::cout << "Blocked and Unrolled Performance = " << calculateGFLOPS(ms_double.count())
              << " GFLOPS/s\n";
    writeMatrixToCSV(result, "data/blockedUnrolled.csv", R1, C2);

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
    writeMatrixToCSV(result, "data/openblas.csv", R1, C2);

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
