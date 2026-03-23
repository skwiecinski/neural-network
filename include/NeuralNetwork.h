#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>

#include "Layer.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include "NetworkConfig.h" 

template <FloatingPoint T>
class NeuralNetwork {
public:
    NeuralNetwork(const NetworkConfig& config);

    NeuralNetwork();

    void addLayer(size_t outputSize, const std::string& activationName);

    Matrix<T> predict(const Matrix<T>& inputData) const;

    void save(const std::string& directoryPath) const;
    void load(const std::string& directoryPath);

    std::vector<Matrix<T>> forwardPass(const Matrix<T>& inputData);
    void backwardPass(const std::vector<Matrix<T>>& layerInputs, const Matrix<T>& lossGradient);
    void updateParameters(); 

    const LossFunction<T>& getLossFunction() const { return *lossFunction; }
    const std::string& getLossFunctionName() const { return lossFunctionName; }
    const std::vector<std::unique_ptr<Layer<T>>>& getLayers() const { return layers; }
    T getLearningRate() const { return learningRate; }

    const NetworkConfig& getConfig() const { return config; }

private:
    std::vector<std::unique_ptr<Layer<T>>> layers;
    T learningRate;
    std::unique_ptr<LossFunction<T>> lossFunction;
    std::string lossFunctionName;

    NetworkConfig config; 
};

#endif 