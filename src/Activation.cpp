#include "Activation.h"

template <FloatingPoint T>
Matrix<T> Activation<T>::sigmoidApply(const Matrix<T>& input) {
    return input.apply_function([](T x) {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::sigmoidDerivative(const Matrix<T>& input) {
    Matrix<T> sigmoidOutput = sigmoidApply(input);
    return sigmoidOutput.hadamardProduct(
        sigmoidOutput.apply_function([](T xOut) {
            return static_cast<T>(1.0) - xOut;
            })
    );
}

template <FloatingPoint T>
Matrix<T> Activation<T>::reluApply(const Matrix<T>& input) {
    return input.apply_function([](T x) {
        return std::max(static_cast<T>(0.0), x);
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::reluDerivative(const Matrix<T>& input) {
    return input.apply_function([](T x) {
        return (x > static_cast<T>(0.0)) ? static_cast<T>(1.0) : static_cast<T>(0.0);
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::leakyReluApply(const Matrix<T>& input) {
    const T alpha = static_cast<T>(0.01);
    return input.apply_function([alpha](T x) {
        return std::max(alpha * x, x);
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::leakyReluDerivative(const Matrix<T>& input) {
    const T alpha = static_cast<T>(0.01);
    return input.apply_function([alpha](T x) {
        return (x > static_cast<T>(0.0)) ? static_cast<T>(1.0) : alpha;
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::tanhApply(const Matrix<T>& input) {
    return input.apply_function([](T x) {
        return std::tanh(x);
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::tanhDerivative(const Matrix<T>& input) {
    Matrix<T> tanhOutput = tanhApply(input);
    return tanhOutput.apply_function([](T xOut) {
        return static_cast<T>(1.0) - xOut * xOut;
        });
}

template <FloatingPoint T>
Matrix<T> Activation<T>::softmaxApply(const Matrix<T>& input) {
    Matrix<T> result(input.getRows(), input.getCols());

    for (size_t i = 0; i < input.getRows(); ++i) {
        T maxVal = input(i, 0);
        for (size_t j = 1; j < input.getCols(); ++j) {
            if (input(i, j) > maxVal) {
                maxVal = input(i, j);
            }
        }

        T sumExp = 0.0;
        for (size_t j = 0; j < input.getCols(); ++j) {
            sumExp += std::exp(input(i, j) - maxVal);
        }

        for (size_t j = 0; j < input.getCols(); ++j) {
            result(i, j) = std::exp(input(i, j) - maxVal) / sumExp;
        }
    }
    return result;
}

template <FloatingPoint T>
Matrix<T> Activation<T>::softmaxDerivative(const Matrix<T>& input) {
    Matrix<T> softmaxOutput = softmaxApply(input);
    return softmaxOutput.hadamardProduct(
        softmaxOutput.apply_function([](T xOut) {
            return static_cast<T>(1.0) - xOut;
            })
    );
}

template <FloatingPoint T>
Activation<T>::Activation(const std::string& activationName) : name(activationName) {
    if (name != "sigmoid" && name != "relu" && name != "leaky_relu" && name != "tanh" && name != "softmax") {
        throw std::invalid_argument("Unknown activation function: " + activationName);
    }
}

template <FloatingPoint T>
Matrix<T> Activation<T>::apply(const Matrix<T>& input) const {
    if (name == "sigmoid") {
        return sigmoidApply(input);
    }
    else if (name == "relu") {
        return reluApply(input);
    }
    else if (name == "leaky_relu") {
        return leakyReluApply(input);
    }
    else if (name == "tanh") {
        return tanhApply(input);
    }
    else if (name == "softmax") {
        return softmaxApply(input);
    }
    else {
        throw std::runtime_error("Invalid activation function state during apply.");
    }
}

template <FloatingPoint T>
Matrix<T> Activation<T>::derivative(const Matrix<T>& input) const {
    if (name == "sigmoid") {
        return sigmoidDerivative(input);
    }
    else if (name == "relu") {
        return reluDerivative(input);
    }
    else if (name == "leaky_relu") {
        return leakyReluDerivative(input);
    }
    else if (name == "tanh") {
        return tanhDerivative(input);
    }
    else if (name == "softmax") {
        return softmaxDerivative(input);
    }
    else {
        throw std::runtime_error("Invalid activation function state during derivative calculation.");
    }
}

template class Activation<double>;
template class Activation<float>;