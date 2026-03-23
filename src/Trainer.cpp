#include <iomanip> 
#include <chrono>
#include <filesystem>

#include "Trainer.h"
#include "Metrics.h"

namespace fs = std::filesystem;

std::vector<double> trainingLossHistory;
std::vector<double> validationAccuracyHistory;

template <FloatingPoint T>
Trainer<T>::Trainer(NeuralNetwork<T>& network, DataLoader<T>& trainDataLoader, std::optional<DataLoader<T>*> validationDataLoader)
    : network(network), trainDataLoader(trainDataLoader), validationDataLoader(validationDataLoader) {
}

template <FloatingPoint T>
void Trainer<T>::runTrainingLoop(size_t epochs) {
    std::cout << "Starting training for " << epochs << " epochs...\n";
    std::cout << std::fixed << std::setprecision(4);

    trainingLossHistory.clear();
    validationAccuracyHistory.clear();

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        trainDataLoader.shuffle();
        trainDataLoader.resetBatchIterator();

        T totalLoss = static_cast<T>(0.0);
        size_t numBatches = 0;

        while (true) {
            std::optional<std::pair<Matrix<T>, Matrix<T>>> batch = trainDataLoader.nextBatch();
            if (!batch.has_value()) {
                break;
            }

            Matrix<T> inputBatch = batch->first;
            Matrix<T> targetBatch = batch->second;

            inputBatch = inputBatch.apply_function([](T pixelValue) {
                return pixelValue / static_cast<T>(255.0);
                });

            std::vector<Matrix<T>> layerInputs = network.forwardPass(inputBatch);
            Matrix<T> predictions = layerInputs.back();

            T batchLoss = network.getLossFunction().calculateLoss(predictions, targetBatch);
            totalLoss += batchLoss;
            numBatches++;

            Matrix<T> lossGradient = network.getLossFunction().calculateGradient(predictions, targetBatch);

            network.backwardPass(layerInputs, lossGradient);
            network.updateParameters();
        }

        T averageTrainLoss = (numBatches > 0) ? totalLoss / numBatches : static_cast<T>(0.0);
        trainingLossHistory.push_back(averageTrainLoss);

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Train Loss: " << averageTrainLoss;

        if (validationDataLoader.has_value()) {
            T validationAccuracy = validate();
            validationAccuracyHistory.push_back(validationAccuracy);
            std::cout << " - Validation Accuracy: " << validationAccuracy * static_cast<T>(100.0) << "%\n";
        }
        else {
            std::cout << "\n";
        }
    }
    std::cout << "Training finished.\n";
}

template <FloatingPoint T>
T Trainer<T>::validate() {
    if (!validationDataLoader.has_value()) {
        throw std::runtime_error("Validation DataLoader not provided to Trainer.");
    }

    DataLoader<T>& currentValidationDataLoader = *(validationDataLoader.value());
    currentValidationDataLoader.resetBatchIterator();

    return Metrics<T>::computeAccuracy(network, currentValidationDataLoader);
}

template <FloatingPoint T>
T Trainer<T>::calculateAverageLoss(DataLoader<T>& dataLoader) {
    dataLoader.resetBatchIterator();
    T totalLoss = static_cast<T>(0.0);
    size_t numBatches = 0;

    while (true) {
        std::optional<std::pair<Matrix<T>, Matrix<T>>> batch = dataLoader.nextBatch();
        if (!batch.has_value()) {
            break;
        }

        Matrix<T> inputBatch = batch->first;
        Matrix<T> targetBatch = batch->second;

        inputBatch = inputBatch.apply_function([](T pixelValue) {
            return pixelValue / static_cast<T>(255.0);
            });

        Matrix<T> predictions = network.predict(inputBatch);
        totalLoss += network.getLossFunction().calculateLoss(predictions, targetBatch);
        numBatches++;
    }
    return (numBatches > 0) ? totalLoss / numBatches : static_cast<T>(0.0);
}

template class Trainer<double>;
template class Trainer<float>;