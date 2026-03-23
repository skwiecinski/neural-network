# Neural Network implementation - Handwritten Digit Recognition

A high-performance Multi-Layer Perceptron (MLP) implementation built from scratch in C++20, featuring NVIDIA CUDA acceleration for the MNIST handwritten digit dataset.

## Description
This project is a full-scale neural network designed to recognize handwritten digits. It focuses on fundamental machine learning algorithms and performance optimization through hardware acceleration on both CPU and GPU.

### Key Features
* Heterogeneous Computing: Supports matrix operations on both CPU (via std::thread and std::async) and GPU (via NVIDIA CUDA and cuBLAS).
* Custom Matrix Engine: A dedicated Matrix class manages linear algebra operations with data stored in row-major order.
* Modular Architecture: Support for multiple activation functions (ReLU, Sigmoid, Softmax, Tanh, Leaky ReLU) and loss functions (MSE, Cross-Entropy).
* Interactive SFML Mode: A graphical interface allowing users to draw digits and receive real-time predictions.
* Persistence & Security: Model serialization to JSON for configuration and CSV for weights, featuring atomic saving via .tmp files.
* User Management: A system for registration and login with masked password input to manage personalized model directories.

## Getting Started

### Dependencies
* C++20 Compiler: Required for features like Concepts (FloatingPoint) and std::filesystem.
* SFML 3.0: Used for the interactive drawing window and event handling.
* NVIDIA CUDA Toolkit & cuBLAS: Essential for high-performance matrix multiplication on the GPU.
* nlohmann/json: Used for serializing and deserializing network configurations.
* Gnuplot: Required to generate training history charts for loss and accuracy.

### Installing
* Clone the repository:
```bash
git clone [https://github.com/skwiecinski/neural-network.git](https://github.com/skwiecinski/neural-network.git)
```
* Environment Setup: Ensure SFML 3.0 and CUDA Toolkit are correctly configured in your system PATH.
*  Data: Place the MNIST dataset files in the data/MNIST/ directory as defined in the global settings.

### Executing Program
The application is managed through a centralized console control panel:
*  Authentication: Login or register to access your personal model directory.
*  Configuration: Initialize a new network and select the computation mode (CPU or CUDA).
*  Training: Run the training loop to train the network on the MNIST dataset.
*  Drawing Mode: Use the SFML interface to draw digits and test accuracy in real-time.

## Authors
Szymon Kwieciński
[@skwiecinski](https://github.com/skwiecinski)

## License
This project is licensed under the MIT License.

## Acknowledgments
* MNIST Database: For providing the standard handwritten digit dataset.
* SFML Team: For the Simple and Fast Multimedia Library.
* nlohmann: For the JSON for Modern C++ library.
