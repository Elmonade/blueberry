#ifndef READ_H
#define READ_H

#include <string>

double readDoubleFromCSV(const std::string &filename, double* arr, int num_doubles);
void writeMatrixToCSV(const double* matrix, const char* filename, int rows, int cols);

#endif
