#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <optional>
#include <cstddef>
#include <limits> 

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>

#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Globals.h"


void clearConsole();
void pressAnyKeyToContinue();
std::string readPasswordHidden(); 

Matrix<double> convertSfImageToMatrix(const sf::Image& image);
void drawSquare(sf::Image& image, int x, int y, int brushSize, const sf::Color& color);
void runSFMLMode(NeuralNetwork<double>& nn);

extern std::vector<double> trainingLossHistory;
extern std::vector<double> validationAccuracyHistory;
void plotTrainingHistory(const std::string& userModelPath); 

#endif