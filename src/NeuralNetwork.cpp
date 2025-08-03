#include "NeuralNetwork.hpp"
#include "Loss.hpp"
#include "Activation.hpp"

NeuralNetwork::NeuralNetwork() {}

void NeuralNetwork::addLayer(Layer& layer){
    layers.push_back(&layer);
}

Eigen::VectorXf NeuralNetwork::feedfoward(Eigen::VectorXf input){
    Eigen::VectorXf curr_output = input;

    for(Layer* curr_layer: layers){
        curr_output = curr_layer->forward(curr_output);
    }

    return curr_output;
}

void NeuralNetwork::backpropagate(const Eigen::VectorXf& input, const Eigen::VectorXf& actual_output){

    // initialize
    Eigen::VectorXf pred_output = feedfoward(input);
    gradients.clear();
    gradients.resize(layers.size());

    // last layer
    Layer* last_layer = layers.back();
    Eigen::VectorXf delta = Loss::mean_squared_error_derivative(pred_output, actual_output).array() \
                            * Activation::sigmoid_derivative(last_layer->last_z).array();
    
    Eigen::MatrixXf grad_weights_last = delta * last_layer->last_input.transpose();
    Eigen::VectorXf grad_biases_last = delta;
    gradients[layers.size() - 1] = {grad_weights_last, grad_biases_last};
    
    // propagate error back
    for (int i = layers.size() - 2; i >= 0; --i){
        Layer* curr_layer = layers[i];
        Layer* next_layer = layers[i+1];

        delta = (next_layer->weights.transpose() * delta).array() \
                * Activation::sigmoid_derivative(curr_layer->last_z).array();
        
        Eigen::MatrixXf grad_weights = delta * curr_layer->last_input.transpose();
        Eigen::VectorXf grad_biases = delta;
        gradients[i] = {grad_weights, grad_biases};
    }

    
}

