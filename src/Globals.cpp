#include "Globals.h"

const size_t MNIST_IMAGE_SIZE = 28;
const size_t MNIST_INPUT_FEATURES = MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE;
const size_t MNIST_OUTPUT_CLASSES = 10;

const std::string TRAIN_IMAGES_PATH = "data/MNIST/train-images.idx3-ubyte";
const std::string TRAIN_LABELS_PATH = "data/MNIST/train-labels.idx1-ubyte";
const std::string TEST_IMAGES_PATH = "data/MNIST/t10k-images.idx3-ubyte";
const std::string TEST_LABELS_PATH = "data/MNIST/t10k-labels.idx1-ubyte";

const size_t WINDOW_SCALE = 20;
const size_t DRAWING_AREA_SIZE = MNIST_IMAGE_SIZE * WINDOW_SCALE;
const size_t TEXT_AREA_WIDTH = 450;
const size_t WINDOW_WIDTH = DRAWING_AREA_SIZE + TEXT_AREA_WIDTH;
const size_t WINDOW_HEIGHT = DRAWING_AREA_SIZE;