#include <type_traits>
#include <stdexcept> 
#include <vector>
#include <iostream> 
#include <algorithm> 
#include <functional> 
#include <thread>     
#include <future>     

#include "Matrix.h"
#include "CudaMatrixOps.h" 

template <FloatingPoint T>
MultiplicationMode Matrix<T>::currentMultiplicationMode = MultiplicationMode::CPU_THREADS;

template <FloatingPoint T>
void Matrix<T>::setMultiplicationMode(MultiplicationMode mode) {
    currentMultiplicationMode = mode;
}

template <FloatingPoint T>
MultiplicationMode Matrix<T>::getCurrentMultiplicationMode() {
    return currentMultiplicationMode;
}

template <FloatingPoint T>
Matrix<T>::Matrix() : rows(0), cols(0), data() {}

template <FloatingPoint T>
Matrix<T>::Matrix(size_t rows_arg, size_t cols_arg)
    : rows(rows_arg), cols(cols_arg), data(rows_arg* cols_arg) {
    if (rows_arg == 0 || cols_arg == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be zero.");
    }
}

template <FloatingPoint T>
Matrix<T>::Matrix(size_t rows_arg, size_t cols_arg, T initialValue)
    : rows(rows_arg), cols(cols_arg), data(rows_arg* cols_arg, initialValue) {
    if (rows_arg == 0 || cols_arg == 0) {
        throw std::invalid_argument("Matrix dimensions cannot be zero.");
    }
}

template <FloatingPoint T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& data_arg) {
    if (data_arg.empty() || data_arg[0].empty()) {
        throw std::invalid_argument("Input vector of vectors cannot be empty.");
    }
    rows = data_arg.size();
    cols = data_arg[0].size();
    data.resize(rows * cols);

    for (size_t i = 0; i < rows; ++i) {
        if (data_arg[i].size() != cols) {
            throw std::invalid_argument("Rows in input vector of vectors must have consistent sizes.");
        }
        for (size_t j = 0; j < cols; ++j) {
            data[i * cols + j] = data_arg[i][j];
        }
    }
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    Matrix<T> result(rows, cols);

    if (currentMultiplicationMode == MultiplicationMode::CPU_THREADS) {
        std::transform(data.begin(), data.end(), other.data.begin(), result.data.begin(), std::plus<T>());
    }
    else if (currentMultiplicationMode == MultiplicationMode::CUDA_GPU) {
        if constexpr (std::is_same_v<T, double>) {
            try {
                cudaAddMatrices(data.data(), other.data.data(), result.data.data(), static_cast<int>(rows * cols));
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA error in operator+: " << e.what() << "\n";
                std::cerr << "Switching to CPU mode.\n";
                currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
                return (*this) + other;
            }
        }
        else {
            std::cerr << "CUDA addition is currently only implemented for 'double'. Switching to CPU mode.\n";
            currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
            return (*this) + other;
        }
    }
    else {
        throw std::runtime_error("Undefined matrix operation mode.");
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix<T> result(rows, cols);

    if (currentMultiplicationMode == MultiplicationMode::CPU_THREADS) {
        std::transform(data.begin(), data.end(), other.data.begin(), result.data.begin(), std::minus<T>());
    }
    else if (currentMultiplicationMode == MultiplicationMode::CUDA_GPU) {
        if constexpr (std::is_same_v<T, double>) {
            try {
                cudaSubtractMatrices(data.data(), other.data.data(), result.data.data(), static_cast<int>(rows * cols));
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA error in operator-: " << e.what() << "\n";
                std::cerr << "Switching to CPU mode.\n";
                currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
                return (*this) - other;
            }
        }
        else {
            std::cerr << "CUDA subtraction is currently only implemented for 'double'. Switching to CPU mode.\n";
            currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
            return (*this) - other;
        }
    }
    else {
        throw std::runtime_error("Undefined matrix operation mode.");
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::operator*(T scalar) const {
    Matrix<T> result(rows, cols);

    if (currentMultiplicationMode == MultiplicationMode::CPU_THREADS) {
        std::transform(data.begin(), data.end(), result.data.begin(), [scalar](T val) { return val * scalar; });
    }
    else if (currentMultiplicationMode == MultiplicationMode::CUDA_GPU) {
        if constexpr (std::is_same_v<T, double>) {
            try {
                cudaScalarMultiplyMatrix(data.data(), scalar, result.data.data(), static_cast<int>(rows * cols));
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA error in scalar multiplication: " << e.what() << "\n";
                std::cerr << "Switching to CPU mode.\n";
                currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
                return (*this) * scalar;
            }
        }
        else {
            std::cerr << "CUDA scalar multiplication is currently only implemented for 'double'. Switching to CPU mode.\n";
            currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
            return (*this) * scalar;
        }
    }
    else {
        throw std::runtime_error("Undefined matrix operation mode.");
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix for multiplication.");
    }

    Matrix<T> result(rows, other.cols, static_cast<T>(0.0));

    if (currentMultiplicationMode == MultiplicationMode::CPU_THREADS) {
        const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
        std::vector<std::future<void>> futures;

        size_t rowsPerThread = rows / numThreads;
        size_t remainingRows = rows % numThreads;

        for (unsigned int t = 0; t < numThreads; ++t) {
            size_t currentStartRow = t * rowsPerThread;
            size_t currentEndRow = currentStartRow + rowsPerThread + (t == numThreads - 1 ? remainingRows : 0);

            futures.push_back(std::async(std::launch::async, [this, &other, &result, currentStartRow, currentEndRow]() {
                for (size_t i = currentStartRow; i < currentEndRow; ++i) {
                    for (size_t j = 0; j < other.cols; ++j) {
                        T sum = static_cast<T>(0.0);
                        for (size_t k = 0; k < cols; ++k) {
                            sum += (*this)(i, k) * other(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
                }));
        }

        for (auto& f : futures) {
            f.get();
        }
    }
    else if (currentMultiplicationMode == MultiplicationMode::CUDA_GPU) {
        if constexpr (std::is_same_v<T, double>) {
            try {
                cudaMatrixMultiplyCublas(data.data(), static_cast<int>(rows), static_cast<int>(cols),
                    other.data.data(), static_cast<int>(other.rows), static_cast<int>(other.cols),
                    result.data.data());
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA error in operator*: " << e.what() << "\n";
                std::cerr << "Automatically falling back to CPU multiplication (std::thread).\n";
                currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
                return (*this) * other;
            }
        }
        else {
            std::cerr << "CUDA multiplication (cuBLAS) currently only supports 'double' type. Automatically falling back to CPU.\n";
            currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
            return (*this) * other;
        }
    }
    else {
        throw std::runtime_error("Unknown matrix multiplication mode.");
    }

    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::hadamardProduct(const Matrix<T>& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for Hadamard product.");
    }
    Matrix<T> result(rows, cols);

    if (currentMultiplicationMode == MultiplicationMode::CPU_THREADS) {
        std::transform(data.begin(), data.end(), other.data.begin(), result.data.begin(), std::multiplies<T>());
    }
    else if (currentMultiplicationMode == MultiplicationMode::CUDA_GPU) {
        if constexpr (std::is_same_v<T, double>) {
            try {
                cudaHadamardProductMatrices(data.data(), other.data.data(), result.data.data(), static_cast<int>(rows * cols));
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA error in Hadamard Product: " << e.what() << "\n";
                std::cerr << "Automatically falling back to CPU.\n";
                currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
                return hadamardProduct(other);
            }
        }
        else {
            std::cerr << "CUDA Hadamard product is currently only implemented for 'double'. Automatically falling back to CPU.\n";
            currentMultiplicationMode = MultiplicationMode::CPU_THREADS;
            return hadamardProduct(other);
        }
    }
    else {
        throw std::runtime_error("Unknown matrix operation mode.");
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Matrix<T>::apply_function(const std::function<T(T)>& func) const {
    Matrix<T> result(rows, cols);
    std::transform(data.begin(), data.end(), result.data.begin(), func);
    return result;
}

template <FloatingPoint T>
T& Matrix<T>::operator()(size_t row_idx, size_t col_idx) {
    if (row_idx >= rows || col_idx >= cols) {
        throw std::out_of_range("Matrix index out of bounds.");
    }
    return data[row_idx * cols + col_idx];
}

template <FloatingPoint T>
const T& Matrix<T>::operator()(size_t row_idx, size_t col_idx) const {
    if (row_idx >= rows || col_idx >= cols) {
        throw std::out_of_range("Matrix index out of bounds.");
    }
    return data[row_idx * cols + col_idx];
}

template <FloatingPoint T>
void Matrix<T>::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

template class Matrix<double>;
template class Matrix<float>;