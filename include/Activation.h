#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>
#include <cmath>    
#include <algorithm>
#include <stdexcept> 

#include "Matrix.h"

template <FloatingPoint T>
class Activation {
public:
    Activation(const std::string& activationName);

    Matrix<T> apply(const Matrix<T>& input) const;

    Matrix<T> derivative(const Matrix<T>& input) const;

private:
    std::string name;

    // Sigmoid: f(x) = 1 / (1 + e^-x)
    static Matrix<T> sigmoidApply(const Matrix<T>& input);
    // Sigmoid Derivative: f'(x) = f(x) * (1 - f(x))
    static Matrix<T> sigmoidDerivative(const Matrix<T>& input);

    // ReLU: f(x) = max(0, x)
    static Matrix<T> reluApply(const Matrix<T>& input);
    // ReLU Derivative: f'(x) = 1 if x > 0 else 0
    static Matrix<T> reluDerivative(const Matrix<T>& input);

    // Leaky ReLU: f(x) = max(alpha * x, x)
    static Matrix<T> leakyReluApply(const Matrix<T>& input);
    // Leaky ReLU Derivative: f'(x) = 1 if x > 0 else alpha
    static Matrix<T> leakyReluDerivative(const Matrix<T>& input);

    // Tanh: f(x) = tanh(x)
    static Matrix<T> tanhApply(const Matrix<T>& input);
    // Tanh Derivative: f'(x) = 1 - f(x)^2
    static Matrix<T> tanhDerivative(const Matrix<T>& input);

    // Softmax: f(x_i) = e^x_i / sum(e^x_j)
    static Matrix<T> softmaxApply(const Matrix<T>& input);
    // Softmax Derivative (element-wise for context of Activation class): y * (1 - y)
    static Matrix<T> softmaxDerivative(const Matrix<T>& input);
};

#endif 