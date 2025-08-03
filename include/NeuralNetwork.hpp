#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include "Eigen/Dense"
#include "Layer.hpp"

class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork();

    // Adds a layer to the network
    void addLayer(const Layer& layer);

    // Forward pass through the network
    // input: input vector for the network
    Eigen::VectorXf predict(Eigen::VectorXf input);

private:
    std::vector<Layer> layers;
};


#endif