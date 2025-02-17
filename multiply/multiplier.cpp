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

class Multiplier {
public:
  const int mat1[R1 * C1];
  const int mat2[R2 * C2];
  const int result[R1 * C2];

  virtual void run(int mat1, int mat2, int result) = 0;
  double time() {
    auto t1 = high_resolution_clock::now();
    run(mat1, mat2, result);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << "Normal Time = " << ms_double.count() << "ms\n";

    for (int i = 0; i < 5; i++) {
      cout << result[i] << " ";
    }
    cout << "\n";
  };
  int read() {
    if (readIntegersFromCSV("multiply/2048x1024.csv", mat1, R1 * C1) !=
        R1 * C1) {
      cout << "Error reading matrix 1\n";
      return 1;
    }

    if (readIntegersFromCSV("multiply/2048x1024.csv", mat2, R2 * C2) !=
        R2 * C2) {
      cout << "Error reading matrix 2\n";
      return 1;
    }

    if (C1 != R2) {
      cout << "Matrix 1 columns doesn't match Matrix 2 rows.\n";
      exit(EXIT_FAILURE);
    }
  }
};
