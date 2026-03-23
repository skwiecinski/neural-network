#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <sstream>

#include "NeuralNetwork.h"

namespace fs = std::filesystem;

template <FloatingPoint T>
NeuralNetwork<T>::NeuralNetwork() :
    learningRate(static_cast<T>(0.0)),
    lossFunction(nullptr),
    lossFunctionName(""),
    config()
{
}

template <FloatingPoint T>
NeuralNetwork<T>::NeuralNetwork(const NetworkConfig& netConfig) :
    config(netConfig), 
    learningRate(netConfig.learningRate), 
    lossFunctionName(netConfig.lossFunctionName),
    lossFunction(nullptr) 
{
    if (config.lossFunctionName == "mse") {
        lossFunction = std::make_unique<MeanSquaredError<T>>();
    }
    else if (config.lossFunctionName == "cross_entropy") {
        lossFunction = std::make_unique<CrossEntropyLoss<T>>();
    }
    else {
        throw std::invalid_argument("Unknown loss function: " + config.lossFunctionName);
    }

    for (const auto& layerConf : config.layers) {
        size_t currentInputSize;
        if (layers.empty()) {
            currentInputSize = config.inputFeatures;
        }
        else {
            currentInputSize = layers.back()->getOutputSize();
        }
        layers.push_back(std::make_unique<Layer<T>>(currentInputSize, layerConf.outputSize, layerConf.activationName));
    }
}

template <FloatingPoint T>
void NeuralNetwork<T>::addLayer(size_t layerOutputSize, const std::string& activationName) {
    size_t currentInputSize;
    if (layers.empty()) {
        currentInputSize = config.inputFeatures;
    }
    else {
        currentInputSize = layers.back()->getOutputSize();
    }
    layers.push_back(std::make_unique<Layer<T>>(currentInputSize, layerOutputSize, activationName));

    if (config.layers.size() < layers.size()) {
        config.layers.emplace_back(layerOutputSize, activationName);
        config.outputClasses = layerOutputSize;
    }
}

template <FloatingPoint T>
std::vector<Matrix<T>> NeuralNetwork<T>::forwardPass(const Matrix<T>& inputData) {
    std::vector<Matrix<T>> layerInputsAndActivations;
    layerInputsAndActivations.push_back(inputData);

    Matrix<T> currentOutput = inputData;
    for (const auto& layer : layers) {
        currentOutput = layer->forward(currentOutput);
        layerInputsAndActivations.push_back(currentOutput);
    }
    return layerInputsAndActivations;
}

template <FloatingPoint T>
void NeuralNetwork<T>::backwardPass(const std::vector<Matrix<T>>& layerInputsAndActivations, const Matrix<T>& lossGradient) {
    Matrix<T> currentError = lossGradient;

    for (size_t i = layers.size(); i-- > 0;) {
        currentError = layers[i]->backward(layerInputsAndActivations[i], currentError);
    }
}

template <FloatingPoint T>
void NeuralNetwork<T>::updateParameters() {
    for (const auto& layer : layers) {
        layer->updateParameters(learningRate);
    }
}

template <FloatingPoint T>
Matrix<T> NeuralNetwork<T>::predict(const Matrix<T>& inputData) const {
    Matrix<T> currentOutput = inputData;
    for (const auto& layer : layers) {
        currentOutput = layer->forward(currentOutput);
    }
    return currentOutput;
}

template <FloatingPoint T>
void NeuralNetwork<T>::save(const std::string& directoryPath) const {
    if (!fs::exists(directoryPath)) {
        fs::create_directories(directoryPath);
    }

    std::string tempConfigFileName = directoryPath + "/config.json.tmp";
    std::string finalConfigFileName = directoryPath + "/config.json";

    std::ofstream configFile(tempConfigFileName);
    if (!configFile.is_open()) {
        throw std::runtime_error("Error: Could not open temporary config file for saving: " + tempConfigFileName);
    }
    NetworkConfig currentConfigToSave = config; 
    currentConfigToSave.multiplicationModeName = Matrix<double>::getCurrentMultiplicationMode() == MultiplicationMode::CPU_THREADS ? "CPU_THREADS" : "CUDA_GPU";
    configFile << currentConfigToSave.toJson().dump(4); 
    configFile.close();

    std::vector<std::string> tempWeightsFiles;
    std::vector<std::string> tempBiasesFiles;

    for (size_t i = 0; i < layers.size(); ++i) {
        std::string tempWeightsFileName = directoryPath + "/layer_" + std::to_string(i) + "_weights.csv.tmp";
        std::ofstream weightsFile(tempWeightsFileName);
        if (!weightsFile.is_open()) {
            fs::remove(tempConfigFileName);
            for (const auto& f : tempWeightsFiles) fs::remove(f);
            for (const auto& f : tempBiasesFiles) fs::remove(f);
            throw std::runtime_error("Error: Could not open temporary file for saving weights: " + tempWeightsFileName);
        }
        weightsFile << std::fixed << std::setprecision(10);
        for (size_t r = 0; r < layers[i]->getWeights().getRows(); ++r) {
            for (size_t c = 0; c < layers[i]->getWeights().getCols(); ++c) {
                weightsFile << layers[i]->getWeights()(r, c) << (c == layers[i]->getWeights().getCols() - 1 ? "" : ",");
            }
            weightsFile << "\n";
        }
        weightsFile.close();
        tempWeightsFiles.push_back(tempWeightsFileName); 

        std::string tempBiasesFileName = directoryPath + "/layer_" + std::to_string(i) + "_biases.csv.tmp";
        std::ofstream biasesFile(tempBiasesFileName);
        if (!biasesFile.is_open()) {
            fs::remove(tempConfigFileName);
            for (const auto& f : tempWeightsFiles) fs::remove(f);
            for (const auto& f : tempBiasesFiles) fs::remove(f);
            throw std::runtime_error("Error: Could not open temporary file for saving biases: " + tempBiasesFileName);
        }
        biasesFile << std::fixed << std::setprecision(10);
        for (size_t r = 0; r < layers[i]->getBiases().getRows(); ++r) {
            for (size_t c = 0; c < layers[i]->getBiases().getCols(); ++c) {
                biasesFile << layers[i]->getBiases()(r, c) << (c == layers[i]->getBiases().getCols() - 1 ? "" : ",");
            }
            biasesFile << "\n";
        }
        biasesFile.close();
        tempBiasesFiles.push_back(tempBiasesFileName); 
    }

    fs::remove(finalConfigFileName);
    for (size_t i = 0; i < layers.size(); ++i) {
        fs::remove(directoryPath + "/layer_" + std::to_string(i) + "_weights.csv");
        fs::remove(directoryPath + "/layer_" + std::to_string(i) + "_biases.csv");
    }

    fs::rename(tempConfigFileName, finalConfigFileName);
    for (size_t i = 0; i < layers.size(); ++i) {
        fs::rename(tempWeightsFiles[i], directoryPath + "/layer_" + std::to_string(i) + "_weights.csv");
        fs::rename(tempBiasesFiles[i], directoryPath + "/layer_" + std::to_string(i) + "_biases.csv");
    }

    std::cout << "Network saved successfully to directory: " << directoryPath << "\n";
}

template <FloatingPoint T>
void NeuralNetwork<T>::load(const std::string& directoryPath) {
    if (!fs::exists(directoryPath)) {
        throw std::runtime_error("Error: Directory does not exist: " + directoryPath);
    }

    std::string configFileName = directoryPath + "/config.json";
    std::ifstream configFile(configFileName);
    if (!configFile.is_open()) {
        throw std::runtime_error("Error: Could not open config file for loading: " + configFileName);
    }

    nlohmann::json j_config;
    try {
        configFile >> j_config;
    }
    catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Error: Failed to parse config.json in " + directoryPath + ": " + e.what());
    }
    configFile.close();

    NetworkConfig loadedConfig = NetworkConfig::fromJson(j_config);

    layers.clear(); 
    lossFunction.reset(); 

    config = loadedConfig; 
    learningRate = config.learningRate;
    lossFunctionName = config.lossFunctionName;

    if (config.lossFunctionName == "mse") {
        lossFunction = std::make_unique<MeanSquaredError<T>>();
    }
    else if (config.lossFunctionName == "cross_entropy") {
        lossFunction = std::make_unique<CrossEntropyLoss<T>>();
    }
    else {
        throw std::invalid_argument("Unknown loss function: " + config.lossFunctionName);
    }

    for (const auto& layerConf : config.layers) {
        size_t currentInputSize;
        if (layers.empty()) {
            currentInputSize = config.inputFeatures;
        }
        else {
            currentInputSize = layers.back()->getOutputSize();
        }
        layers.push_back(std::make_unique<Layer<T>>(currentInputSize, layerConf.outputSize, layerConf.activationName));
    }

    if (config.multiplicationModeName == "CPU_THREADS") {
        Matrix<double>::setMultiplicationMode(MultiplicationMode::CPU_THREADS);
    }
    else if (config.multiplicationModeName == "CUDA_GPU") {
        Matrix<double>::setMultiplicationMode(MultiplicationMode::CUDA_GPU);
    }
    else {
        std::cerr << "Warning: Unknown multiplication mode '" << config.multiplicationModeName << "' in config. Defaulting to CPU.\n";
        Matrix<double>::setMultiplicationMode(MultiplicationMode::CPU_THREADS);
    }

    if (layers.size() != config.layers.size()) {
        throw std::runtime_error("Internal error: Mismatch between loaded config layers and created layers count during load.");
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        std::string weightsFileName = directoryPath + "/layer_" + std::to_string(i) + "_weights.csv";
        std::ifstream weightsFile(weightsFileName);
        if (!weightsFile.is_open()) {
            throw std::runtime_error("Error: Could not open file for loading weights: " + weightsFileName + ". Model might be incomplete or corrupted.");
        }

        size_t r_idx = 0;
        std::string dataLine;
        while (std::getline(weightsFile, dataLine) && r_idx < layers[i]->getWeights().getRows()) {
            std::stringstream ss(dataLine);
            std::string segment;
            size_t c_idx = 0;
            while (std::getline(ss, segment, ',') && c_idx < layers[i]->getWeights().getCols()) {
                try {
                    layers[i]->getWeights()(r_idx, c_idx) = static_cast<T>(std::stod(segment));
                }
                catch (const std::invalid_argument& e) {
                    weightsFile.close();
                    throw std::runtime_error("Invalid number format in weights file: " + weightsFileName + " at row " + std::to_string(r_idx + 1) + ", col " + std::to_string(c_idx + 1) + ": " + e.what());
                }
                catch (const std::out_of_range& e) {
                    weightsFile.close();
                    throw std::runtime_error("Number out of range in weights file: " + weightsFileName + " at row " + std::to_string(r_idx + 1) + ", col " + std::to_string(c_idx + 1) + ": " + e.what());
                }
                c_idx++;
            }
            if (c_idx != layers[i]->getWeights().getCols()) {
                throw std::runtime_error("Error: Incorrect number of weight columns in file: " + weightsFileName + " at row " + std::to_string(r_idx + 1));
            }
            r_idx++;
        }
        if (r_idx != layers[i]->getWeights().getRows()) {
            throw std::runtime_error("Error: Incorrect number of weight rows in file: " + weightsFileName);
        }
        weightsFile.close();

        std::string biasesFileName = directoryPath + "/layer_" + std::to_string(i) + "_biases.csv";
        std::ifstream biasesFile(biasesFileName);
        if (!biasesFile.is_open()) {
            throw std::runtime_error("Error: Could not open file for loading biases: " + biasesFileName + ". Model might be incomplete or corrupted.");
        }

        r_idx = 0;
        while (std::getline(biasesFile, dataLine) && r_idx < layers[i]->getBiases().getRows()) {
            std::stringstream ss(dataLine);
            std::string segment;
            size_t c_idx = 0;
            while (std::getline(ss, segment, ',') && c_idx < layers[i]->getBiases().getCols()) {
                try {
                    layers[i]->getBiases()(r_idx, c_idx) = static_cast<T>(std::stod(segment));
                }
                catch (const std::invalid_argument& e) {
                    biasesFile.close();
                    throw std::runtime_error("Invalid number format in biases file: " + biasesFileName + " at row " + std::to_string(r_idx + 1) + ", col " + std::to_string(c_idx + 1) + ": " + e.what());
                }
                catch (const std::out_of_range& e) {
                    biasesFile.close();
                    throw std::runtime_error("Number out of range in biases file: " + biasesFileName + " at row " + std::to_string(r_idx + 1) + ", col " + std::to_string(c_idx + 1) + ": " + e.what());
                }
                c_idx++;
            }
            if (c_idx != layers[i]->getBiases().getCols()) {
                throw std::runtime_error("Error: Incorrect number of columns for biases in file: " + biasesFileName + " at row " + std::to_string(r_idx + 1));
            }
            r_idx++;
        }
        if (r_idx != layers[i]->getBiases().getRows()) {
            throw std::runtime_error("Error: Incorrect number of rows for biases in file: " + biasesFileName);
        }
        biasesFile.close();
    }
    std::cout << "Network loaded successfully from directory: " << directoryPath << "\n";
}

template class NeuralNetwork<double>;
template class NeuralNetwork<float>;