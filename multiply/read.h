#ifndef READ_H
#define READ_H

#include <string>

int readIntegersFromCSV(const std::string &filename, int* arr, int num_integers);
void writeMatrixToCSV(const int* matrix, const char* filename, int rows, int cols);

#endif
