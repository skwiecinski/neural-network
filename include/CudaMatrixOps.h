#ifndef CUDA_MATRIX_OPS_H
#define CUDA_MATRIX_OPS_H

extern "C" void cudaMatrixMultiplyCublas(const double* h_A, int numRowsA, int numColsA,
    const double* h_B, int numRowsB, int numColsB,
    double* h_C);

extern "C" void cudaAddMatrices(const double* h_A, const double* h_B, double* h_C, int numElements);
extern "C" void cudaSubtractMatrices(const double* h_A, const double* h_B, double* h_C, int numElements);
extern "C" void cudaScalarMultiplyMatrix(const double* h_A, double scalar, double* h_C, int numElements);
extern "C" void cudaHadamardProductMatrices(const double* h_A, const double* h_B, double* h_C, int numElements);


#endif 