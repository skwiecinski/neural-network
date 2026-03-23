# Neural Network implementation - Handwritten Digit Recognition

A high-performance Multi-Layer Perceptron (MLP) implementation built from scratch in C++20, featuring NVIDIA CUDA acceleration for the MNIST handwritten digit dataset.

## Description
This project is a full-scale neural network designed to recognize handwritten digits. It focuses on fundamental machine learning algorithms and performance optimization through hardware acceleration on both CPU and GPU. The system includes a custom math engine, interactive drawing mode, and a multi-user management system.

## Technical Implementation & Best Practices
* **Modern C++20 Standards:** Leverages the latest features including Concepts for template constraints (FloatingPoint), std::filesystem for robust I/O, and std::async for parallel CPU processing.
* **Clean Code & Architecture:** Adheres to clean code principles with a strictly modular, object-oriented design. Each class (Matrix, Layer, NeuralNetwork) has a single, clear responsibility.
* **Memory Safety:** High-level memory management using RAII and smart pointers (std::unique_ptr), ensuring zero memory leaks and robust resource handling.
* **Heterogeneous Computing:** Optimized kernels for NVIDIA GPUs using CUDA and cuBLAS, with a fallback to multi-threaded CPU execution.

## Key Features
* **Custom Matrix Engine:** Dedicated Matrix<T> class managing linear algebra operations in row-major order.
* **Modular Architecture:** Support for various activation functions (ReLU, Sigmoid, Softmax, Tanh, Leaky ReLU) and loss functions (MSE, Cross-Entropy).
* **Interactive SFML Mode:** Real-time digit recognition through a graphical drawing interface.
* **Persistence:** Atomic model saving (using .tmp files) to prevent data corruption, with configuration stored in JSON and weights in CSV.
* **User System:** Secure registration/login system with masked password input.

## Getting Started

### Dependencies
* C++20 Compatible Compiler (GCC 11+, Clang 13+, or MSVC 19.29+).
* SFML 3.0 (Simple and Fast Multimedia Library).
* NVIDIA CUDA Toolkit & cuBLAS.
* nlohmann/json (included via headers).
* Gnuplot (optional, for training history visualization).

### Building the Project
Windows:
*  It is highly recommended to use the "Developer Command Prompt for VS 2022" to ensure cl.exe is in your PATH for the CUDA compiler (nvcc).
*  Set your SFML_PATH and CUDA_PATH in the Makefile or provide them during build:
```bash
make SFML_PATH="C:/SFML-3.0"
```
Linux:
*  Ensure SFML 3.0 and CUDA are installed in standard paths (/usr/local).
*  Run: 
```bash
make
```
### Installing
*  Clone the repository:
```bash
git clone [https://github.com/skwiecinski/neural-network.git](https://github.com/skwiecinski/neural-network.git)
```
*  Data: Place the MNIST dataset files in the 'data/MNIST/' directory.

## Authors
Szymon Kwieciński
[@skwiecinski](https://github.com/skwiecinski)

## License
This project is licensed under the MIT License.

## Acknowledgments
* MNIST Database: For the handwritten digit dataset.
* SFML Team: For the multimedia library.
* nlohmann: For the JSON for Modern C++ library.
