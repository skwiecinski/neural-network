#include <thread>
#include <future> 

#include "DataLoader.h"
#include "Globals.h"

template <FloatingPoint T>
DataLoader<T>::DataLoader(const std::string& imagePath, const std::string& labelPath, size_t batchSize)
    : imagePath(imagePath), labelPath(labelPath), batchSize(batchSize), currentBatchIdx(0) {
    if (batchSize == 0) {
        throw std::invalid_argument("Batch size cannot be zero.");
    }
}

template <FloatingPoint T>
void DataLoader<T>::loadData() {
    loadImageData(imagePath);
    loadLabelData(labelPath);

    if (imageMatrix.getRows() != labelMatrix.getRows()) {
        throw std::runtime_error("Number of images and labels do not match.");
    }

    sampleIndices.resize(imageMatrix.getRows());
    std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
}

template <FloatingPoint T>
uint32_t DataLoader<T>::reverseEndian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
        ((val << 8) & 0x00FF0000) |
        ((val >> 8) & 0x0000FF00) |
        ((val >> 24) & 0x000000FF);
}

template <FloatingPoint T>
void DataLoader<T>::loadImageData(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + path);
    }

    uint32_t magicNumber = 0;
    uint32_t numImages = 0;
    uint32_t numRows = 0;
    uint32_t numCols = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseEndian(magicNumber);
    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number.");
    }

    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    numImages = reverseEndian(numImages);
    file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
    numRows = reverseEndian(numRows);
    file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
    numCols = reverseEndian(numCols);

    imageMatrix = Matrix<T>(numImages, numRows * numCols);

    const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::future<void>> futures;

    std::vector<unsigned char> rawPixels(static_cast<size_t>(numImages) * numRows * numCols);
    file.read(reinterpret_cast<char*>(rawPixels.data()), rawPixels.size());
    file.close();

    size_t imagesPerThread = numImages / numThreads;
    size_t remainingImages = numImages % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t startImage = t * imagesPerThread;
        size_t endImage = startImage + imagesPerThread + (t == numThreads - 1 ? remainingImages : 0);

        futures.push_back(std::async(std::launch::async,
            [this, &rawPixels, startImage, endImage, numRows, numCols]() {
                size_t pixelsPerImage = static_cast<size_t>(numRows) * numCols;
                for (size_t i = startImage; i < endImage; ++i) {
                    for (size_t j = 0; j < pixelsPerImage; ++j) {
                        imageMatrix(i, j) = static_cast<T>(rawPixels[i * pixelsPerImage + j]);
                    }
                }
            }));
    }
    for (auto& f : futures) {
        f.get();
    }
}

template <FloatingPoint T>
void DataLoader<T>::loadLabelData(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + path);
    }

    uint32_t magicNumber = 0;
    uint32_t numLabels = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = reverseEndian(magicNumber);
    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number.");
    }

    file.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
    numLabels = reverseEndian(numLabels);

    labelMatrix = Matrix<T>(numLabels, 10);

    std::vector<unsigned char> rawLabels(numLabels);
    file.read(reinterpret_cast<char*>(rawLabels.data()), rawLabels.size());
    file.close();

    const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::future<void>> futures;

    size_t labelsPerThread = numLabels / numThreads;
    size_t remainingLabels = numLabels % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t startLabel = t * labelsPerThread;
        size_t endLabel = startLabel + labelsPerThread + (t == numThreads - 1 ? remainingLabels : 0);

        futures.push_back(std::async(std::launch::async,
            [this, &rawLabels, startLabel, endLabel]() {
                for (size_t i = startLabel; i < endLabel; ++i) {
                    unsigned char label = rawLabels[i];
                    for (size_t j = 0; j < 10; ++j) {
                        labelMatrix(i, j) = static_cast<T>(0.0);
                    }
                    if (label >= 0 && label < 10) {
                        labelMatrix(i, static_cast<size_t>(label)) = static_cast<T>(1.0);
                    }
                    else {
                        throw std::runtime_error("Invalid label value during parallel loading.");
                    }
                }
            }));
    }
    for (auto& f : futures) {
        try {
            f.get();
        }
        catch (const std::exception& e) {
            std::cerr << "Error in loading labels: " << e.what() << "\n";
            throw; 
        }
    }
}

template <FloatingPoint T>
void DataLoader<T>::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sampleIndices.begin(), sampleIndices.end(), g);
    resetBatchIterator();
}

template <FloatingPoint T>
std::optional<std::pair<Matrix<T>, Matrix<T>>> DataLoader<T>::nextBatch() {
    if (currentBatchIdx >= sampleIndices.size()) {
        return std::nullopt;
    }

    size_t batchStart = currentBatchIdx;
    size_t batchEnd = std::min(currentBatchIdx + batchSize, sampleIndices.size());
    size_t actualBatchSize = batchEnd - batchStart;

    if (actualBatchSize == 0) {
        return std::nullopt;
    }

    Matrix<T> batchImages(actualBatchSize, imageMatrix.getCols());
    Matrix<T> batchLabels(actualBatchSize, labelMatrix.getCols());

    const unsigned int numThreads = std::thread::hardware_concurrency() != 0 ? std::thread::hardware_concurrency() : 4;
    std::vector<std::future<void>> futures;

    size_t rowsPerThread = actualBatchSize / numThreads;
    size_t remainingRows = actualBatchSize % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t startRow = t * rowsPerThread;
        size_t endRow = startRow + rowsPerThread + (t == numThreads - 1 ? remainingRows : 0);

        futures.push_back(std::async(std::launch::async,
            [this, &batchImages, &batchLabels, startRow, endRow, batchStart]() {
                for (size_t i = startRow; i < endRow; ++i) {
                    size_t originalIndex = sampleIndices[batchStart + i];
                    for (size_t j = 0; j < imageMatrix.getCols(); ++j) {
                        batchImages(i, j) = imageMatrix(originalIndex, j);
                    }
                    for (size_t j = 0; j < labelMatrix.getCols(); ++j) {
                        batchLabels(i, j) = labelMatrix(originalIndex, j);
                    }
                }
            }));
    }
    for (auto& f : futures) {
        f.get();
    }

    currentBatchIdx = batchEnd;
    return std::make_optional(std::make_pair(batchImages, batchLabels));
}

template <FloatingPoint T>
void DataLoader<T>::resetBatchIterator() {
    currentBatchIdx = 0;
}

template class DataLoader<double>;
template class DataLoader<float>;