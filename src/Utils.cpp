#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <limits> 
#include <algorithm>
#include <random>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

#include "Utils.h"
#include "Globals.h"
#include "Matrix.h"
#include "NeuralNetwork.h"

namespace fs = std::filesystem;

void clearConsole() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void pressAnyKeyToContinue() {
    std::cout << "\nPress any key to continue...";
    std::cin.ignore(10000, '\n'); 
    std::cin.get();
}

std::string readPasswordHidden() {
    std::string password;
#ifdef _WIN32
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hStdin, &mode);
    SetConsoleMode(hStdin, mode & (~ENABLE_ECHO_INPUT));

    char ch;
    while ((ch = _getch()) != '\r') {
        if (ch == '\b') {
            if (!password.empty()) {
                password.pop_back();
                std::cout << "\b \b";
            }
        }
        else if (ch >= 32 && ch <= 126) {
            password.push_back(ch);
            std::cout << "*";
        }
    }
    SetConsoleMode(hStdin, mode);
    std::cout << std::endl;
#else
    termios oldt;
    tcgetattr(STDIN_FILENO, &oldt);
    termios newt = oldt;
    newt.c_lflag &= ~ECHO;
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    char ch;
    while (std::cin.get(ch) && ch != '\n') {
        if (ch == 127 || ch == '\b') {
            if (!password.empty()) {
                password.pop_back();
                std::cout << "\b \b";
            }
        }
        else if (ch >= 32 && ch <= 126) {
            password.push_back(ch);
            std::cout << "*";
        }
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    std::cout << std::endl;
#endif
    return password;
}

Matrix<double> convertSfImageToMatrix(const sf::Image& image) {
    Matrix<double> result(1, MNIST_INPUT_FEATURES);

    for (size_t y = 0; y < MNIST_IMAGE_SIZE; ++y) {
        for (size_t x = 0; x < MNIST_IMAGE_SIZE; ++x) {
            sf::Color color = image.getPixel({ static_cast<unsigned int>(x * WINDOW_SCALE), static_cast<unsigned int>(y * WINDOW_SCALE) });
            double pixelValue = static_cast<double>(color.r) / 255.0;
            result(0, y * MNIST_IMAGE_SIZE + x) = pixelValue;
        }
    }
    return result;
}

void drawSquare(sf::Image& image, int x, int y, int brushSize, const sf::Color& color) {
    for (int i = -brushSize / 2; i <= brushSize / 2; ++i) {
        for (int j = -brushSize / 2; j <= brushSize / 2; ++j) {
            int drawX = x + i;
            int drawY = y + j;
            if (drawX >= 0 && drawX < (int)DRAWING_AREA_SIZE && drawY >= 0 && drawY < (int)DRAWING_AREA_SIZE) {
                image.setPixel({ static_cast<unsigned int>(drawX), static_cast<unsigned int>(drawY) }, color);
            }
        }
    }
}

void updatePredictionText(NeuralNetwork<double>& nn, const sf::Image& drawingImage, sf::Text& predictionText) {
    if (nn.getLayers().empty()) {
        predictionText.setString("Error: Network not configured!");
        predictionText.setFillColor(sf::Color::Red);
        return;
    }
    Matrix<double> inputForNN = convertSfImageToMatrix(drawingImage);
    Matrix<double> predictions = nn.predict(inputForNN);

    size_t predictedDigit = 0;
    double maxProb = -1.0;
    for (size_t i = 0; i < MNIST_OUTPUT_CLASSES; ++i) {
        if (predictions(0, i) > maxProb) {
            maxProb = predictions(0, i);
            predictedDigit = i;
        }
    }

    std::stringstream ss;
    ss << "Guessed: " << predictedDigit << "\n"
        << "Prob: " << std::fixed << std::setprecision(2) << maxProb * 100.0 << "%";
    predictionText.setString(ss.str());
    predictionText.setFillColor(sf::Color::Green);
}


void runSFMLMode(NeuralNetwork<double>& nn) {
    sf::RenderWindow window(sf::VideoMode({ static_cast<unsigned int>(WINDOW_WIDTH), static_cast<unsigned int>(WINDOW_HEIGHT) }), "Draw number");
    window.setFramerateLimit(60);

    sf::Image drawingImage({ static_cast<unsigned int>(DRAWING_AREA_SIZE), static_cast<unsigned int>(DRAWING_AREA_SIZE) }, sf::Color::Black);

    sf::Texture drawingTexture;
    drawingTexture.loadFromImage(drawingImage);
    sf::Sprite drawingSprite(drawingTexture);

    int brushSize = WINDOW_SCALE * 2;

    sf::Font font;
    std::optional<sf::Font> loadedFont = sf::Font{ "data/assets/arial.ttf" };
    if (!loadedFont.has_value()) {
        std::cerr << "Cannot load font.\n";
        return;
    }
    font = *loadedFont;

    sf::Text predictionText(font, sf::String("Guessed: "), 36);
    predictionText.setFillColor(sf::Color::Red);
    predictionText.setPosition({ static_cast<float>(DRAWING_AREA_SIZE + 20), 20.f });

    sf::Text instructionText(font, sf::String("Draw with mouse.\nLeft button: Draw.\nRight button: Erase.\nSpace: Guess.\nC: Clear."), 20);
    instructionText.setFillColor(sf::Color(200, 200, 200));
    instructionText.setPosition({ static_cast<float>(DRAWING_AREA_SIZE + 20), static_cast<float>(WINDOW_HEIGHT - instructionText.getGlobalBounds().size.y - 20) });


    bool drawing = false;
    bool erasing = false;
    bool imageChanged = true;

    while (window.isOpen()) {
        std::optional<sf::Event> eventOpt;
        while (eventOpt = window.pollEvent()) {
            sf::Event event = *eventOpt;

            if (event.is<sf::Event::Closed>()) {
                window.close();
            }
            else if (auto* mouseButtonPressed = event.getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    drawing = true;
                    erasing = false;
                }
                else if (mouseButtonPressed->button == sf::Mouse::Button::Right) {
                    erasing = true;
                    drawing = false;
                }
            }
            else if (auto* mouseButtonReleased = event.getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left || mouseButtonReleased->button == sf::Mouse::Button::Right) {
                    drawing = false;
                    erasing = false;
                }
            }
            else if (auto* mouseMoved = event.getIf<sf::Event::MouseMoved>()) {
                if (mouseMoved->position.x >= 0 && mouseMoved->position.x < (int)DRAWING_AREA_SIZE &&
                    mouseMoved->position.y >= 0 && mouseMoved->position.y < (int)DRAWING_AREA_SIZE)
                {
                    if (drawing) {
                        drawSquare(drawingImage, mouseMoved->position.x, mouseMoved->position.y, brushSize, sf::Color::White);
                        drawingTexture.loadFromImage(drawingImage);
                        imageChanged = true;
                    }
                    else if (erasing) {
                        drawSquare(drawingImage, mouseMoved->position.x, mouseMoved->position.y, brushSize, sf::Color::Black);
                        drawingTexture.loadFromImage(drawingImage);
                        imageChanged = true;
                    }
                }
            }
            else if (auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->code == sf::Keyboard::Key::Space) {
                    imageChanged = true;
                }
                else if (keyPressed->code == sf::Keyboard::Key::C) {
                    drawingImage = sf::Image({ static_cast<unsigned int>(DRAWING_AREA_SIZE), static_cast<unsigned int>(DRAWING_AREA_SIZE) }, sf::Color::Black);
                    drawingTexture.loadFromImage(drawingImage);
                    predictionText.setString("Guessed: ");
                    predictionText.setFillColor(sf::Color::White);
                    imageChanged = true;
                }
            }
        }

        if (imageChanged) {
            updatePredictionText(nn, drawingImage, predictionText);
            imageChanged = false;
        }

        window.clear(sf::Color::Black);
        window.draw(drawingSprite);
        window.draw(predictionText);
        window.draw(instructionText);
        window.display();
    }
}

void plotTrainingHistory(const std::string& userModelPath) {
    const std::string resultsDir = userModelPath + "/results";

    if (!fs::exists(resultsDir)) {
        try {
            fs::create_directories(resultsDir);
            std::cout << "Created directory: " << resultsDir << "\n";
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Error: Could not create directory '" << resultsDir << "': " << e.what() << "\n";
            return;
        }
    }

    if (trainingLossHistory.empty()) {
        std::cout << "No data to draw a chart.\n";
        return;
    }

    std::ofstream lossFile(resultsDir + "/training_loss.dat");
    if (!lossFile.is_open()) {
        std::cerr << "Error: Cannot open " << resultsDir << "/training_loss.dat to save.\n";
        return;
    }
    for (size_t i = 0; i < trainingLossHistory.size(); ++i) {
        lossFile << i + 1 << " " << trainingLossHistory[i] << "\n";
    }
    lossFile.close();

    std::ofstream accuracyFile(resultsDir + "/validation_accuracy.dat");
    if (!accuracyFile.is_open()) {
        std::cerr << "Error: Cannot open " << resultsDir << "/validation_accuracy.dat to save.\n";
        return;
    }
    for (size_t i = 0; i < validationAccuracyHistory.size(); ++i) {
        accuracyFile << i + 1 << " " << validationAccuracyHistory[i] << "\n";
    }
    accuracyFile.close();

    std::string gnuplotCommands =
        "set title 'Neural Network Training History'\n"
        "set xlabel 'Epoch'\n"
        "set ylabel 'Value'\n"
        "set grid\n"
        "set key autotitle columnhead\n"
        "set term pngcairo size 1000,700\n"
        "set output '" + resultsDir + "/training_history.png'\n\n" + 
        "plot '" + resultsDir + "/training_loss.dat' using 1:2 with lines title 'Average Training Loss' linecolor rgb 'red', \\\n" +
        "     '" + resultsDir + "/validation_accuracy.dat' using 1:2 with lines title 'Validation Accuracy' linecolor rgb 'blue'\n";

    std::ofstream gnuplotScript(resultsDir + "/plot_script.gp");
    if (!gnuplotScript.is_open()) {
        std::cerr << "Error: Could not create file " << resultsDir << "/plot_script.gp.\n";
        return;
    }
    gnuplotScript << gnuplotCommands;
    gnuplotScript.close();

    std::string systemCommand = "gnuplot \"" + resultsDir + "/plot_script.gp\"";
    int result = system(systemCommand.c_str());
    if (result != 0) {
        std::cerr << "Error: Could not execute 'gnuplot'. Ensure it is installed and in PATH.\n";
    }
    else {
        std::cout << "Training history plot saved to '" << resultsDir << "/training_history.png'.\n";
    }

#ifdef _WIN32
    system(("start " + resultsDir + "/training_history.png").c_str());
#elif __APPLE__
    system(("open " + resultsDir + "/training_history.png").c_str());
#else
    system(("xdg-open " + resultsDir + "/training_history.png").c_str());
#endif
}