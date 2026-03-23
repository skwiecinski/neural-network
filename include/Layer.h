#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include <random>

#include "Matrix.h"
#include "Activation.h"

template <FloatingPoint T>
class Layer {
public:
    Layer(size_t inputSize, size_t outputSize, const std::string& activationName);

    Matrix<T> forward(const Matrix<T>& inputData);

    Matrix<T> backward(const Matrix<T>& inputDataForThisLayer, const Matrix<T>& errorFromNextLayer);

    const Matrix<T>& getWeightsGradients() const { return weightsGradients; }
    const Matrix<T>& getBiasesGradients() const { return biasesGradients; }

    void updateParameters(T learningRate);

    size_t getInputSize() const { return weights.getRows(); }
    size_t getOutputSize() const { return weights.getCols(); }
    const Matrix<T>& getPreActivations() const { return preActivations; }
    const Matrix<T>& getActivations() const { return activations; }
    Matrix<T>& getWeights() { return weights; } 
    Matrix<T>& getBiases() { return biases; } 
    const std::string& getActivationFunctionName() const { return activationFunctionName; }

private:
    Matrix<T> weights;
    Matrix<T> biases;
    Matrix<T> activations;
    Matrix<T> preActivations;

    Matrix<T> weightsGradients;
    Matrix<T> biasesGradients;

    Activation<T> activationFunction;
    std::string activationFunctionName;

    void initializeWeights(size_t inputSize, size_t outputSize);
};

#endif 