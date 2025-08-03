#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::addLayer(const Layer& layer){
    layers.push_back(layer);
}

Eigen::VectorXf NeuralNetwork::predict(Eigen::VectorXf input){
    Eigen::VectorXf curr_output = input;

    for(Layer& curr_layer: layers){
        curr_output = curr_layer.forward(curr_output);
    }

    return curr_output;
}

