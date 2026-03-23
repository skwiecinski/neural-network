#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <optional> 

#include "Matrix.h"

template <FloatingPoint T>
class DataLoader {
public:
    DataLoader(const std::string& imagePath, const std::string& labelPath, size_t batchSize);

    void loadData();
    void shuffle();
    std::optional<std::pair<Matrix<T>, Matrix<T>>> nextBatch();
    void resetBatchIterator();

    size_t getNumSamples() const { return imageMatrix.getRows(); }
    size_t getNumFeatures() const { return imageMatrix.getCols(); }
    size_t getNumClasses() const { return labelMatrix.getCols(); }
    size_t getBatchSize() const { return batchSize; }

private:
    std::string imagePath;
    std::string labelPath;
    size_t batchSize;

    Matrix<T> imageMatrix;
    Matrix<T> labelMatrix;

    std::vector<size_t> sampleIndices;
    size_t currentBatchIdx;

    static uint32_t reverseEndian(uint32_t val);
    void loadImageData(const std::string& path);
    void loadLabelData(const std::string& path);
};

#endif