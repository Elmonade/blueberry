// read.h
#ifndef READ_H
#define READ_H

#include <string>
#include <vector>

int readIntegersFromCSV(const std::string &filename, std::vector<int> &matrix, 
                        int rows, int cols);
#endif
