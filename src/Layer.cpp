#include <random>

#include "Layer.h"

template <FloatingPoint T>
Layer<T>::Layer(size_t inputSize, size_t outputSize, const std::string& activationName)
    : weights(inputSize, outputSize),
    biases(1, outputSize, static_cast<T>(0.0)),
    weightsGradients(inputSize, outputSize, static_cast<T>(0.0)),
    biasesGradients(1, outputSize, static_cast<T>(0.0)),
    activationFunction(activationName),
    activationFunctionName(activationName) { 
    initializeWeights(inputSize, outputSize);
}

template <FloatingPoint T>
void Layer<T>::initializeWeights(size_t inputSize, size_t outputSize) {
    std::random_device rd;
    std::mt19937 generator(rd());

    T stdDev = static_cast<T>(std::sqrt(static_cast<T>(2.0) / inputSize));
    std::normal_distribution<T> distribution(static_cast<T>(0.0), stdDev);

    for (size_t i = 0; i < weights.getRows(); ++i) {
        for (size_t j = 0; j < weights.getCols(); ++j) {
            weights(i, j) = distribution(generator);
        }
    }
}
template <FloatingPoint T>
Matrix<T> Layer<T>::forward(const Matrix<T>& inputData) {
    preActivations = inputData * weights;

    const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::future<void>> futures;

    size_t rowsPerThread = preActivations.getRows() / numThreads;
    size_t remainingRows = preActivations.getRows() % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t startRow = t * rowsPerThread;
        size_t endRow = startRow + rowsPerThread + (t == numThreads - 1 ? remainingRows : 0);

        futures.push_back(std::async(std::launch::async, [this, startRow, endRow]() {
            for (size_t i = startRow; i < endRow; ++i) {
                for (size_t j = 0; j < this->preActivations.getCols(); ++j) { 
                    this->preActivations(i, j) = this->preActivations(i, j) + this->biases(0, j);
                }
            }
            }));
    }

    for (auto& f : futures) {
        f.get();
    }

    activations = activationFunction.apply(preActivations);

    return activations;
}

template <FloatingPoint T>
Matrix<T> Layer<T>::backward(const Matrix<T>& inputDataForThisLayer, const Matrix<T>& errorFromNextLayer) {
    Matrix<T> dActivation = activationFunction.derivative(preActivations);

    Matrix<T> currentLayerError = errorFromNextLayer.hadamardProduct(dActivation);

    biasesGradients = Matrix<T>(1, biases.getCols(), static_cast<T>(0.0)); 

    const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::future<void>> futures;

    size_t colsPerThread = biases.getCols() / numThreads;
    size_t remainingCols = biases.getCols() % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t startCol = t * colsPerThread;
        size_t endCol = startCol + colsPerThread + (t == numThreads - 1 ? remainingCols : 0);

        futures.push_back(std::async(std::launch::async, [this, &currentLayerError, startCol, endCol]() {
            for (size_t col = startCol; col < endCol; ++col) {
                T sum = static_cast<T>(0.0);
                for (size_t row = 0; row < currentLayerError.getRows(); ++row) {
                    sum += currentLayerError(row, col);
                }
                biasesGradients(0, col) = sum;
            }
            }));
    }
    for (auto& f : futures) {
        f.get();
    }

    weightsGradients = inputDataForThisLayer.transpose() * currentLayerError;

    Matrix<T> errorToPreviousLayer = weights * currentLayerError.transpose();

    errorToPreviousLayer = errorToPreviousLayer.transpose();

    return errorToPreviousLayer;
}

template <FloatingPoint T>
void Layer<T>::updateParameters(T learningRate) {
    weights = weights - (weightsGradients * learningRate);
    biases = biases - (biasesGradients * learningRate);
}

template class Layer<double>;
template class Layer<float>;