#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <numeric>
#include <thread>
#include <future>
#include <stdexcept>
#include <algorithm>
#include <concepts>
#include <cmath>
#include <functional>

template <typename T>
concept FloatingPoint = std::floating_point<T>;

enum class MultiplicationMode {
    CPU_THREADS,
    CUDA_GPU
};

template <FloatingPoint T>
class Matrix {
public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, T initialValue);
    Matrix(const std::vector<std::vector<T>>& data);


    Matrix<T> operator+(const Matrix<T>& other) const;
    Matrix<T> operator-(const Matrix<T>& other) const;
    Matrix<T> operator*(const Matrix<T>& other) const; 
    Matrix<T> operator*(T scalar) const;            

    Matrix<T> transpose() const;
    Matrix<T> hadamardProduct(const Matrix<T>& other) const; 

    Matrix<T> apply_function(const std::function<T(T)>& func) const;

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    T& operator()(size_t row, size_t col);
    const T& operator()(size_t row, size_t col) const;
    void print() const;

    static void setMultiplicationMode(MultiplicationMode mode);
    static MultiplicationMode getCurrentMultiplicationMode();

private:
    size_t rows;
    size_t cols;
    std::vector<T> data;

    static MultiplicationMode currentMultiplicationMode;
};

#endif