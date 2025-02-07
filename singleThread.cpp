#include <chrono>
#include <iostream>
#include "read.h"

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2
#define C1 2
#define R2 2
#define C2 3

void mulMat(const int (&mat1)[R1 * C1], const int (&mat2)[R2 * C2]) {
    int result[R1 * C2];
    int cnt = 0;

    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            for (int l = 0; l < R2; l++) {
                result[cnt] = mat1[i + l] * mat2[j + l];
                cout << result[cnt] << "\n";
                cnt++;
            }
        }
    }

    /*
    for (int i = 0; i < R1; i++) {
      for (int j = 0; j < C2; j++) {
        result[i * C2 + j] = 0;
        for (int k = 0; k < C1; k++) {
          result[i * C2 + j] += mat1[i * C1 + k] * mat2[k * C2 + j];
        }
      }
    }
    */

    for (const int i: result) {
        cout << i << "\n";
    }
}

int main() {
    constexpr int mat1[R1 * C1] = {1, 1, 2, 2};
    const int mat2[R2 * C2] = {1, 1, 1, 2, 2, 2};

    if (C1 != R2) {
        cout << "Matrix 1 columns doesn't match Matrix 2 rows.\n";
        exit(EXIT_FAILURE);
    }

    const auto t1 = high_resolution_clock::now();
    mulMat(mat1, mat2);
    const auto t2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    return 0;
}
