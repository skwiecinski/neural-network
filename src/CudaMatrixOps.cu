#include "CudaMatrixOps.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <algorithm> // Dla std::max

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::stringstream ss_cuda_err; \
            ss_cuda_err << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err); \
            throw std::runtime_error(ss_cuda_err.str()); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::stringstream ss_cublas_err; \
            ss_cublas_err << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << " - Status: " << static_cast<int>(status); \
            throw std::runtime_error(ss_cublas_err.str()); \
        } \
    } while (0)

#define TILE_DIM 32 

__global__ void transposeSharedMemKernel(const double* A_rm, double* AT_cm, int R, int K) {
    __shared__ double tile[TILE_DIM][TILE_DIM + 1]; // +1 padding dla unikania bank conflicts

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int r_global_A = blockIdx.y * TILE_DIM + ty;
    int k_global_A = blockIdx.x * TILE_DIM + tx;

    if (r_global_A < R && k_global_A < K) {
        tile[ty][tx] = A_rm[r_global_A * K + k_global_A];
    }
    __syncthreads();

    int r_global_AT = blockIdx.x * TILE_DIM + ty; // Nowy wiersz w AT_cm (oryginalna kolumna k_global_A)
    int k_global_AT = blockIdx.y * TILE_DIM + tx; // Nowa kolumna w AT_cm (oryginalny wiersz r_global_A)

    if (r_global_AT < K && k_global_AT < R) {
        AT_cm[r_global_AT + k_global_AT * K] = tile[tx][ty];
    }
}

extern "C" void cudaMatrixMultiplyCublas(const double* h_A, int numRowsA, int numColsA,
    const double* h_B, int numRowsB, int numColsB,
    double* h_C) {
    if (numColsA != numRowsB) {
        throw std::invalid_argument("CUDA cuBLAS: Wymiary macierzy nie pasuja do mnozenia (numColsA != numRowsB).");
    }

    cublasHandle_t cublasH = nullptr;
    double* d_A_rm = nullptr;
    double* d_B_rm = nullptr;
    double* d_A_cm = nullptr; // (h_A)^T w formacie column-major
    double* d_B_cm = nullptr; // (h_B)^T w formacie column-major
    double* d_C_prod_cm = nullptr; // Wynik (h_A * h_B)^T w formacie column-major

    size_t sizeA_rm = static_cast<size_t>(numRowsA) * numColsA * sizeof(double);
    size_t sizeB_rm = static_cast<size_t>(numRowsB) * numColsB * sizeof(double);
    size_t sizeA_cm = static_cast<size_t>(numColsA) * numRowsA * sizeof(double);
    size_t sizeB_cm = static_cast<size_t>(numColsB) * numRowsB * sizeof(double);
    size_t sizeC_prod_cm = static_cast<size_t>(numColsB) * numRowsA * sizeof(double);
    size_t size_hC = static_cast<size_t>(numRowsA) * numColsB * sizeof(double);

    const double alpha = 1.0;
    const double beta = 0.0;

    try {
        CUBLAS_CHECK(cublasCreate(&cublasH));

        CUDA_CHECK(cudaMalloc((void**)&d_A_rm, sizeA_rm));
        CUDA_CHECK(cudaMalloc((void**)&d_B_rm, sizeB_rm));
        CUDA_CHECK(cudaMalloc((void**)&d_A_cm, sizeA_cm));
        CUDA_CHECK(cudaMalloc((void**)&d_B_cm, sizeB_cm));
        CUDA_CHECK(cudaMalloc((void**)&d_C_prod_cm, sizeC_prod_cm));

        CUDA_CHECK(cudaMemcpy(d_A_rm, h_A, sizeA_rm, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_rm, h_B, sizeB_rm, cudaMemcpyHostToDevice));

        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDimA((numColsA + blockDim.x - 1) / blockDim.x,
            (numRowsA + blockDim.y - 1) / blockDim.y);
        transposeSharedMemKernel << <gridDimA, blockDim >> > (d_A_rm, d_A_cm, numRowsA, numColsA);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 gridDimB((numColsB + blockDim.x - 1) / blockDim.x,
            (numRowsB + blockDim.y - 1) / blockDim.y);
        transposeSharedMemKernel << <gridDimB, blockDim >> > (d_B_rm, d_B_cm, numRowsB, numColsB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_A_rm)); d_A_rm = nullptr;
        CUDA_CHECK(cudaFree(d_B_rm)); d_B_rm = nullptr;


        CUBLAS_CHECK(cublasDgemm(cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            numColsB,   
            numRowsA,    
            numRowsB,    
            &alpha,
            d_B_cm,     
            numColsB,    
            d_A_cm,      
            numColsA,   
            &beta,
            d_C_prod_cm,
            numColsB));  

        CUDA_CHECK(cudaMemcpy(h_C, d_C_prod_cm, size_hC, cudaMemcpyDeviceToHost));

    }
    catch (const std::exception& e) {
        std::cerr << "Error in cudaMatrixMultiplyCublas: " << e.what() << std::endl;
        if (d_A_rm) cudaFree(d_A_rm); 
        if (d_B_rm) cudaFree(d_B_rm);
        if (d_C_prod_cm && h_C == nullptr) cudaFree(d_C_prod_cm); 
        if (d_A_cm) cudaFree(d_A_cm);
        if (d_B_cm) cudaFree(d_B_cm);
        if (cublasH) cublasDestroy(cublasH); 
        throw;
    }

    if (cublasH) CUBLAS_CHECK(cublasDestroy(cublasH));
    if (d_C_prod_cm) CUDA_CHECK(cudaFree(d_C_prod_cm));
    if (d_A_cm) CUDA_CHECK(cudaFree(d_A_cm));
    if (d_B_cm) CUDA_CHECK(cudaFree(d_B_cm));
}

template <typename Op>
__global__ void elementWiseKernel(const double* A, const double* B, double* C, int numElements, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        C[idx] = op(A[idx], B[idx]);
    }
}

__global__ void scalarMultiplyKernel(const double* A, double scalar, double* C, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        C[idx] = A[idx] * scalar;
    }
}

struct AddOp {
    __device__ double operator()(double a, double b) { return a + b; }
};
struct SubtractOp {
    __device__ double operator()(double a, double b) { return a - b; }
};
struct MultiplyOp { // Dla Hadamard product
    __device__ double operator()(double a, double b) { return a * b; }
};

void launchElementWiseKernel(const double* h_A, const double* h_B, double* h_C, int numElements, const std::string& opType) {
    double* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
    size_t size = static_cast<size_t>(numElements) * sizeof(double);

    // Symulacja bloku finally dla zwolnienia pami�ci
    auto cleanup = [&]() {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        };

    try {
        CUDA_CHECK(cudaMalloc((void**)&d_A, size));
        CUDA_CHECK(cudaMalloc((void**)&d_B, size));
        CUDA_CHECK(cudaMalloc((void**)&d_C, size));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
        if (h_B) { 
            CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
        }


        const int BLOCK_SIZE = 256;
        dim3 dimGrid(static_cast<unsigned int>((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE));


        if (opType == "add") {
            elementWiseKernel << <dimGrid, BLOCK_SIZE >> > (d_A, d_B, d_C, numElements, AddOp());
        }
        else if (opType == "subtract") {
            elementWiseKernel << <dimGrid, BLOCK_SIZE >> > (d_A, d_B, d_C, numElements, SubtractOp());
        }
        else if (opType == "hadamard") {
            elementWiseKernel << <dimGrid, BLOCK_SIZE >> > (d_A, d_B, d_C, numElements, MultiplyOp());
        }
        else {
            cleanup(); 
            throw std::runtime_error("Nieznany typ operacji element-wise.");
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
        cleanup();
    }
    catch (const std::exception& e) {
        std::cerr << "Wyjatek w launchElementWiseKernel (" << opType << "): " << e.what() << std::endl;
        cleanup(); 
        throw;
    }
}

extern "C" void cudaAddMatrices(const double* h_A, const double* h_B, double* h_C, int numElements) {
    launchElementWiseKernel(h_A, h_B, h_C, numElements, "add");
}
extern "C" void cudaSubtractMatrices(const double* h_A, const double* h_B, double* h_C, int numElements) {
    launchElementWiseKernel(h_A, h_B, h_C, numElements, "subtract");
}
extern "C" void cudaHadamardProductMatrices(const double* h_A, const double* h_B, double* h_C, int numElements) {
    launchElementWiseKernel(h_A, h_B, h_C, numElements, "hadamard");
}

extern "C" void cudaScalarMultiplyMatrix(const double* h_A, double scalar, double* h_C, int numElements) {
    double* d_A = nullptr, * d_C = nullptr;
    size_t size = static_cast<size_t>(numElements) * sizeof(double);

    auto cleanup = [&]() {
        if (d_A) cudaFree(d_A);
        if (d_C) cudaFree(d_C);
        };

    try {
        CUDA_CHECK(cudaMalloc((void**)&d_A, size));
        CUDA_CHECK(cudaMalloc((void**)&d_C, size));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

        const int BLOCK_SIZE = 256;
        dim3 dimGrid(static_cast<unsigned int>((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE));


        scalarMultiplyKernel << <dimGrid, BLOCK_SIZE >> > (d_A, scalar, d_C, numElements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
        cleanup();
    }
    catch (const std::exception& e) {
        std::cerr << "Wyjatek w cudaScalarMultiplyMatrix: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}
