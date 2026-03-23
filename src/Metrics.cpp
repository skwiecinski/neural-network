#include "Metrics.h"
#include "Globals.h"

template <FloatingPoint T>
T Metrics<T>::computeAccuracy(NeuralNetwork<T>& network, DataLoader<T>& dataLoader) {
    size_t correctPredictions = 0;
    size_t totalPredictions = 0;

    dataLoader.resetBatchIterator();

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

        for (size_t i = 0; i < predictions.getRows(); ++i) {
            totalPredictions++;

            size_t predictedClass = 0;
            T maxVal = predictions(i, 0);
            for (size_t j = 1; j < predictions.getCols(); ++j) {
                if (predictions(i, j) > maxVal) {
                    maxVal = predictions(i, j);
                    predictedClass = j;
                }
            }

            size_t trueClass = 0;
            for (size_t j = 0; j < targetBatch.getCols(); ++j) {
                if (targetBatch(i, j) > static_cast<T>(0.5)) {
                    trueClass = j;
                    break;
                }
            }

            if (predictedClass == trueClass) {
                correctPredictions++;
            }
        }
    }
    if (totalPredictions == 0) return static_cast<T>(0.0);
    return static_cast<T>(correctPredictions) / totalPredictions;
}

template class Metrics<double>;
template class Metrics<float>;