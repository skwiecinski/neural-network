#ifndef NETWORK_CONFIG_H
#define NETWORK_CONFIG_H

#include <vector>
#include <string>
#include <cstddef> 

#include "Matrix.h" 
#include "json.hpp" 

struct LayerConfig {
    size_t outputSize;
    std::string activationName;

    LayerConfig(size_t os = 0, const std::string& an = "") : outputSize(os), activationName(an) {}

    nlohmann::json toJson() const {
        return {
            {"outputSize", outputSize},
            {"activationName", activationName}
        };
    }

    static LayerConfig fromJson(const nlohmann::json& j) {
        return LayerConfig(j.at("outputSize").get<size_t>(), j.at("activationName").get<std::string>());
    }
};

struct NetworkConfig {
    size_t inputFeatures;
    size_t outputClasses;
    double learningRate;
    std::string lossFunctionName;
    std::string optimizerName; 
    std::vector<LayerConfig> layers;

    std::string multiplicationModeName; 

    NetworkConfig() : inputFeatures(0), outputClasses(0), learningRate(0.0), lossFunctionName(""), optimizerName("sgd"), multiplicationModeName("CPU_THREADS") {}

    nlohmann::json toJson() const {
        nlohmann::json j;
        j["inputFeatures"] = inputFeatures;
        j["outputClasses"] = outputClasses;
        j["learningRate"] = learningRate;
        j["lossFunctionName"] = lossFunctionName;
        j["multiplicationModeName"] = multiplicationModeName;

        nlohmann::json layers_json = nlohmann::json::array();
        for (const auto& layer : layers) {
            layers_json.push_back(layer.toJson());
        }
        j["layers"] = layers_json;
        return j;
    }

    static NetworkConfig fromJson(const nlohmann::json& j) {
        NetworkConfig config;
        config.inputFeatures = j.at("inputFeatures").get<size_t>();
        config.outputClasses = j.at("outputClasses").get<size_t>();
        config.learningRate = j.at("learningRate").get<double>();
        config.lossFunctionName = j.at("lossFunctionName").get<std::string>();
        config.multiplicationModeName = j.at("multiplicationModeName").get<std::string>();

        for (const auto& layer_j : j.at("layers")) {
            config.layers.push_back(LayerConfig::fromJson(layer_j));
        }
        return config;
    }
};

#endif 