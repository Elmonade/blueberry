#include <chrono>
#include <iostream>
using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define R1 2
#define C1 2
#define R2 2
#define C2 3

void mulMat(int mat1[][C1], int mat2[][C2]) {
    int result[R1][C2];

    for (int i = 0; i < R1; i++) {
        for (int j = 0; j < C2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < R2; k++)
                result[i][j] += mat1[i][k] * mat2[k][j];
            cout << result[i][j] << "\t";
        }
        cout << "\n";
    }
}

int main() {
    int mat1[R1][C1] = {{3, 4}, {5, 6}};
    int mat2[R2][C2] = {{2, 4, 6}, {1, 3, 7}};

    if (C1 != R2) {
        cout << "Matrix 1 columns doesn't match Matrix 2 rows.\n";
        exit(EXIT_FAILURE);
    }

    const auto t1 = high_resolution_clock::now();
    mulMat(mat1, mat2);
    const auto t2 = high_resolution_clock::now();

    /* Milliseconds as a double. */
    const duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
    return 0;
}
