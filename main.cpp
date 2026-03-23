#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <chrono>
#include <optional>

#include "Matrix.h"
#include "Activation.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include "Trainer.h"
#include "Metrics.h"
#include "Utils.h"
#include "Globals.h"
#include "UserManager.h" 

std::string handleAuthentication(UserManager& userManager) {
    std::string currentUsername;
    int authChoice;
    do {
        clearConsole();
        std::cout << "--- AUTHENTICATION PANEL ---\n";
        std::cout << "1. Login\n";
        std::cout << "2. Register\n";
        std::cout << "0. Exit program\n";
        std::cout << "Select option: ";

        std::cin >> authChoice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid choice. Please enter a number.\n";
            pressAnyKeyToContinue();
            continue;
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        std::string username, password;
        switch (authChoice) {
        case 1:
            std::cout << "Enter username: ";
            std::cin >> username;
            std::cout << "Enter password: ";
            password = readPasswordHidden(); 
            if (userManager.loginUser(username, password)) {
                currentUsername = username;
                return currentUsername;
            }
            pressAnyKeyToContinue();
            break;
        case 2: 
            std::cout << "Enter new username: ";
            std::cin >> username;
            std::cout << "Enter new password: ";
            password = readPasswordHidden(); 
            userManager.registerUser(username, password);
            pressAnyKeyToContinue();
            break;
        case 0: 
            return ""; 
        default:
            std::cout << "Unknown option. Please try again.\n";
            pressAnyKeyToContinue();
            break;
        }
    } while (true);
}


int main() {
    Matrix<double>::setMultiplicationMode(MultiplicationMode::CPU_THREADS);

    UserManager userManager;
    std::string loggedInUsername = handleAuthentication(userManager);

    if (loggedInUsername.empty()) {
        std::cout << "Exiting program.\n";
        return 0;
    }

    std::string currentUserModelPath = userManager.getUserModelPath(loggedInUsername);

    std::unique_ptr<NeuralNetwork<double>> nn = nullptr; 
    std::unique_ptr<DataLoader<double>> trainDataLoader = nullptr;
    std::unique_ptr<DataLoader<double>> testDataLoader = nullptr;

    bool networkConfigured = false; 

    int choice;
    do {
        clearConsole();
        std::cout << "--- Neural Network Control Panel ---\n";
        std::cout << "Logged in as: " << loggedInUsername << "\n"; 
        std::cout << "1. Configure new network\n";
        std::cout << "2. Add layer to network\n";
        std::cout << "3. Train network\n";
        std::cout << "4. Save network\n";
        std::cout << "5. Load network\n";
        std::cout << "6. Check accuracy on test data\n";
        std::cout << "7. Draw and guess digit (SFML graphics mode)\n";
        std::cout << "8. Display network structure\n";
        std::cout << "9. Plot training history\n";
        std::cout << "0. Exit\n";
        std::cout << "Select option: ";

        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid choice. Please enter a number.\n";
            pressAnyKeyToContinue();
            continue;
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        try {
            switch (choice) {
            case 1: { 
                clearConsole();
                NetworkConfig newConfig;
                newConfig.inputFeatures = MNIST_INPUT_FEATURES;
                newConfig.outputClasses = MNIST_OUTPUT_CLASSES;
                newConfig.optimizerName = "sgd";

                std::cout << "Enter learning rate (e.g., 0.001): ";
                std::cin >> newConfig.learningRate;
                std::cout << "Enter loss function name (mse/cross_entropy): ";
                std::cin >> newConfig.lossFunctionName;

                std::cout << "\n--- Matrix Multiplication Settings ---\n";
                std::cout << "Select matrix multiplication mode:\n";
                std::cout << "  1. CPU (std::thread)\n";
                std::cout << "  2. CUDA (GPU) - requires NVIDIA card and CUDA Toolkit\n";
                std::cout << "Current mode: " << (Matrix<double>::getCurrentMultiplicationMode() == MultiplicationMode::CPU_THREADS ? "CPU" : "CUDA") << "\n";
                std::cout << "Select option (1 or 2): ";
                int modeChoice;
                std::cin >> modeChoice;

                if (std::cin.fail() || (modeChoice != 1 && modeChoice != 2)) {
                    std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid multiplication mode choice. Defaulting to CPU.\n";
                    Matrix<double>::setMultiplicationMode(MultiplicationMode::CPU_THREADS);
                }
                else if (modeChoice == 1) {
                    Matrix<double>::setMultiplicationMode(MultiplicationMode::CPU_THREADS);
                    std::cout << "Multiplication mode set to CPU (std::thread).\n";
                }
                else if (modeChoice == 2) {
                    Matrix<double>::setMultiplicationMode(MultiplicationMode::CUDA_GPU);
                    std::cout << "Attempting to set multiplication mode to CUDA (GPU).\n";
                    std::cout << "NOTE: Requires NVIDIA graphics card and CUDA Toolkit.\n";
                }

                nn = std::make_unique<NeuralNetwork<double>>(newConfig);
                networkConfigured = true;
                std::cout << "Network initialized.\n";
                pressAnyKeyToContinue();
                break;
            }
            case 2: { 
                if (!networkConfigured) {
                    std::cout << "First, configure the network (option 1).\n";
                    pressAnyKeyToContinue();
                    break;
                }
                size_t outputSize;
                std::string activationName;
                std::cout << "Enter number of neurons in the layer: ";
                std::cin >> outputSize;
                std::cout << "Enter activation function name (relu/sigmoid/tanh/leaky_relu/softmax): ";
                std::cin >> activationName;
                nn->addLayer(outputSize, activationName);
                std::cout << "Layer added. Current number of layers: " << nn->getLayers().size() << "\n";
                pressAnyKeyToContinue();
                break;
            }
            case 3: { 
                if (!networkConfigured || nn->getLayers().empty()) {
                    std::cout << "Network not configured or has no layers.\n";
                    pressAnyKeyToContinue();
                    break;
                }

                size_t trainBatchSize;
                std::cout << "Enter training batch size (e.g., 128): ";
                std::cin >> trainBatchSize;
                if (std::cin.fail() || trainBatchSize == 0) {
                    std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid batch size. Defaulting to 128.\n";
                    trainBatchSize = 128;
                }

                if (!trainDataLoader || trainDataLoader->getBatchSize() != trainBatchSize) {
                    trainDataLoader = std::make_unique<DataLoader<double>>(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, trainBatchSize);
                    trainDataLoader->loadData();
                    std::cout << "Training data loaded with batch_size: " << trainBatchSize << "\n";
                }
                else {
                    std::cout << "Using loaded training data with batch_size: " << trainBatchSize << "\n";
                }


                size_t testBatchSize;
                std::cout << "Enter test batch size (for validation, e.g., 256): ";
                std::cin >> testBatchSize;
                if (std::cin.fail() || testBatchSize == 0) {
                    std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid test batch size. Defaulting to 256.\n";
                    testBatchSize = 256;
                }

                if (!testDataLoader || testDataLoader->getBatchSize() != testBatchSize) {
                    testDataLoader = std::make_unique<DataLoader<double>>(TEST_IMAGES_PATH, TEST_LABELS_PATH, testBatchSize);
                    testDataLoader->loadData();
                    std::cout << "Test data loaded with batch_size: " << testBatchSize << "\n";
                }
                else {
                    std::cout << "Using loaded test data with batch_size: " << testBatchSize << "\n";
                }


                size_t epochs;
                std::cout << "Enter number of training epochs: ";
                std::cin >> epochs;

                trainingLossHistory.clear();
                validationAccuracyHistory.clear();

                Trainer<double> trainer(*nn, *trainDataLoader, testDataLoader.get());
                auto startTime = std::chrono::high_resolution_clock::now();

                trainer.runTrainingLoop(epochs);

                auto endTime = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                std::cout << "\nTraining finished in " << duration.count() / 1000.0 << " seconds.\n";
                pressAnyKeyToContinue();
                break;
            }
            case 4: { 
                if (!networkConfigured || nn->getLayers().empty()) {
                    std::cout << "Network not configured or has no layers to save.\n";
                    pressAnyKeyToContinue();
                    break;
                }
                nn->save(currentUserModelPath); 
                std::cout << "Network saved successfully for user " << loggedInUsername << ".\n";
                pressAnyKeyToContinue();
                break;
            }
            case 5: { 
                nn = std::make_unique<NeuralNetwork<double>>(); 

                nn->load(currentUserModelPath); 

                networkConfigured = true; 
                std::cout << "Network loaded successfully for user " << loggedInUsername << ".\n";
                pressAnyKeyToContinue();
                break;
            }
            case 6: { 
                if (!networkConfigured || nn->getLayers().empty()) {
                    std::cout << "Network not configured or has no layers.\n";
                    pressAnyKeyToContinue();
                    break;
                }

                size_t testBatchSize;
                std::cout << "Enter test batch size (e.g., 256): ";
                std::cin >> testBatchSize;
                if (std::cin.fail() || testBatchSize == 0) {
                    std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    std::cout << "Invalid test batch size. Defaulting to 256.\n";
                    testBatchSize = 256;
                }

                if (!testDataLoader || testDataLoader->getBatchSize() != testBatchSize) {
                    testDataLoader = std::make_unique<DataLoader<double>>(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, testBatchSize);
                    testDataLoader->loadData();
                    std::cout << "Test data loaded with batch_size: " << testBatchSize << "\n";
                }
                else {
                    std::cout << "Using loaded test data with batch_size: " << testBatchSize << "\n";
                }

                double accuracy = Metrics<double>::computeAccuracy(*nn, *testDataLoader);
                std::cout << "Accuracy on test set: " << accuracy * 100.0 << "%\n";
                pressAnyKeyToContinue();
                break;
            }
            case 7: { 
                if (!networkConfigured || nn->getLayers().empty()) {
                    std::cout << "Network not configured or has no layers.\n";
                    std::cout << "Please configure the network (option 1) and add layers (option 2).\n";
                    pressAnyKeyToContinue();
                    break;
                }
                runSFMLMode(*nn);
                break;
            }
            case 8: { 
                if (!networkConfigured) {
                    std::cout << "Network not yet configured.\n";
                    pressAnyKeyToContinue();
                    break;
                }
                std::cout << "\n--- Network Structure ---\n";
                std::cout << "Input size: " << nn->getConfig().inputFeatures << "\n";
                for (size_t i = 0; i < nn->getLayers().size(); ++i) {
                    const auto& layer = nn->getLayers()[i];
                    std::cout << "Layer " << i + 1 << ": "
                        << "Input=" << layer->getInputSize()
                        << ", Output=" << layer->getOutputSize()
                        << ", Activation=" << layer->getActivationFunctionName() << "\n";
                }
                std::cout << "Learning rate: " << nn->getLearningRate() << "\n";
                std::cout << "Loss function: " << nn->getLossFunctionName() << "\n";
                std::cout << "Matrix multiplication mode: " << (Matrix<double>::getCurrentMultiplicationMode() == MultiplicationMode::CPU_THREADS ? "CPU (std::thread)" : "CUDA (GPU)") << "\n";
                pressAnyKeyToContinue();
                break;
            }
            case 9: { 
                plotTrainingHistory(currentUserModelPath);
                pressAnyKeyToContinue();
                break;
            }
            case 0: { 
                std::cout << "Exiting program.\n";
                break;
            }
            default: {
                std::cout << "Unknown option. Please try again.\n";
                pressAnyKeyToContinue();
                break;
            }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "An error occurred: " << e.what() << std::endl;
            pressAnyKeyToContinue();
        }
    } while (choice != 0);

    return 0;
}