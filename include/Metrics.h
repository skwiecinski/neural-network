#ifndef METRICS_H
#define METRICS_H

#include <algorithm> 
#include <vector>

#include "Matrix.h"
#include "DataLoader.h" 
#include "NeuralNetwork.h" 

template <FloatingPoint T>
class Metrics {
public:
    static T computeAccuracy(NeuralNetwork<T>& network, DataLoader<T>& dataLoader);
};

#endif