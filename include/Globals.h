#ifndef GLOBALS_H
#define GLOBALS_H

#include <string>
#include <cstddef> 

extern const size_t MNIST_IMAGE_SIZE;
extern const size_t MNIST_INPUT_FEATURES;
extern const size_t MNIST_OUTPUT_CLASSES;

extern const std::string TRAIN_IMAGES_PATH;
extern const std::string TRAIN_LABELS_PATH;
extern const std::string TEST_IMAGES_PATH;
extern const std::string TEST_LABELS_PATH;

extern const std::string MODEL_SAVE_DIRECTORY;

extern const size_t WINDOW_SCALE;
extern const size_t DRAWING_AREA_SIZE;
extern const size_t TEXT_AREA_WIDTH;
extern const size_t WINDOW_WIDTH;
extern const size_t WINDOW_HEIGHT;

#endif