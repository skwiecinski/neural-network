#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <cmath>   
#include <stdexcept>

#include "Matrix.h"

template <FloatingPoint T>
class LossFunction {
public:
    virtual T calculateLoss(const Matrix<T>& predictions, const Matrix<T>& targets) const = 0;

    virtual Matrix<T> calculateGradient(const Matrix<T>& predictions, const Matrix<T>& targets) const = 0;

    virtual ~LossFunction() = default;
};

template <FloatingPoint T>
class MeanSquaredError : public LossFunction<T> {
public:
    T calculateLoss(const Matrix<T>& predictions, const Matrix<T>& targets) const override {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets must have the same dimensions for MSE.");
        }
        T sumSqDiff = static_cast<T>(0.0); 
        for (size_t i = 0; i < predictions.getRows(); ++i) {
            for (size_t j = 0; j < predictions.getCols(); ++j) {
                sumSqDiff += (std::pow)(predictions(i, j) - targets(i, j), static_cast<T>(2.0)); 
            }
        }
        return sumSqDiff / (predictions.getRows() * predictions.getCols());
    }

    Matrix<T> calculateGradient(const Matrix<T>& predictions, const Matrix<T>& targets) const override {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets must have the same dimensions for MSE gradient calculation.");
        }
        Matrix<T> gradient = predictions - targets;
        return gradient;
    }
};

template <FloatingPoint T>
class CrossEntropyLoss : public LossFunction<T> {
public:
    T calculateLoss(const Matrix<T>& predictions, const Matrix<T>& targets) const override {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets must have the same dimensions for Cross-Entropy Loss.");
        }
        T lossSum = static_cast<T>(0.0); 
        for (size_t i = 0; i < predictions.getRows(); ++i) {
            for (size_t j = 0; j < predictions.getCols(); ++j) {
                T clampedPrediction = (std::max)(predictions(i, j), static_cast<T>(1e-9)); 
                lossSum += targets(i, j) * (std::log)(clampedPrediction); 
            }
        }
        return -lossSum / predictions.getRows();
    }

    Matrix<T> calculateGradient(const Matrix<T>& predictions, const Matrix<T>& targets) const override {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Predictions and targets must have the same dimensions for Cross-Entropy Loss gradient calculation.");
        }
        return predictions - targets;
    }
};

#endif 