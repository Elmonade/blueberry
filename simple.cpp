// C++ program to multiply two matrices
#include <chrono>
#include <iostream>
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::milliseconds;


#define R1 2 // number of rows in Matrix-1
#define C1 2 // number of columns in Matrix-1
#define R2 2 // number of rows in Matrix-2
#define C2 3 // number of columns in Matrix-2

void mulMat(int mat1[][C1], int mat2[][C2]) {
	int rslt[R1][C2];

	for (int i = 0; i < R1; i++) {
		for (int j = 0; j < C2; j++) {
			rslt[i][j] = 0;
			for (int k = 0; k < R2; k++)
				rslt[i][j] += mat1[i][k] * mat2[k][j];
			cout << rslt[i][j] << "\t";
		}
		cout << "\n";
	}
}

// Driver code
int main() {
	// R1 = 4, C1 = 4 and R2 = 4, C2 = 4
	int mat1[R1][C1] = { { 1, 1 }, { 2, 2 } };
	int mat2[R2][C2] = { { 1, 1, 1 }, { 2, 2, 2 } };

	if (C1 != R2) {
		cout << "The number of columns in Matrix-1 must "
				"be equal to the number of rows in "
				"Matrix-2\n";
		cout << "Please update MACROs according to your "
				"array dimension in #define section\n";
		exit(EXIT_FAILURE);
	}

  auto t1 = high_resolution_clock::now();
	mulMat(mat1, mat2);
  auto t2 = high_resolution_clock::now();

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << ms_double.count() << "ms\n";
	return 0;
}
