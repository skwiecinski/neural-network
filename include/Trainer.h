#ifndef TRAINER_H
#define TRAINER_H

#include <iostream>
#include <string>
#include <vector>
#include "json.hpp"

#include "NeuralNetwork.h"
#include "DataLoader.h"
#include "LossFunction.h"

extern std::vector<double> trainingLossHistory;
extern std::vector<double> validationAccuracyHistory;

template <FloatingPoint T>
class Trainer {
public:
    Trainer(NeuralNetwork<T>& network, DataLoader<T>& trainDataLoader, std::optional<DataLoader<T>*> validationDataLoader = std::nullopt);

    void runTrainingLoop(size_t epochs);

    T validate();

private:
    NeuralNetwork<T>& network;
    DataLoader<T>& trainDataLoader;
    std::optional<DataLoader<T>*> validationDataLoader;

    T calculateAverageLoss(DataLoader<T>& dataLoader);
};

#endif