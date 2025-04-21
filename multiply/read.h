#ifndef READ_H
#define READ_H

#include <string>
#include <vector>

struct PlotData {
    std::string label;
    double time;
};

double readDoubleFromCSV(const std::string &filename, double* arr, int num_doubles);
void writeMatrixToCSV(const double* matrix, const char* filename, int rows, int cols);
void writePlotDataToCSV(const std::vector<PlotData>& plotData, const char* filename);
void writeTimingToCSV(const std::string& filename, double timeA, double timeB, double timeC);

#endif
