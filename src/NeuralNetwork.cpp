#include <iostream>
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

void NeuralNetwork::update_weights(double learning_rate){
    if (gradients.size() != layers.size()) {
        std::cout << "Gradients vector size is different than layers vector size!" << std::endl;
        return;
    }

    for(size_t i = 0; i < layers.size(); ++i){
        Layer* current_layer = layers[i];
        const auto& grad_pair = gradients[i]; 


        current_layer->weights -= learning_rate * grad_pair.first;
        current_layer->biases -= learning_rate * grad_pair.second;
    }
}

void NeuralNetwork::train(  
    const std::vector<Eigen::VectorXf>& training_inputs,                     
    const std::vector<Eigen::VectorXf>& training_outputs,
    int epochs,
    double learning_rate
){
    if (training_inputs.size() != training_outputs.size()) {
        std::cout << "Error training: training_inputs.size() != training_outputs.size()" << std::endl;
        return;
    }

    std::cout << "Beginning training..." << std::endl;

    for(int i = 0; i < epochs; ++i){
        double total_epoch_error = 0.0;

        for (size_t j = 0; j < training_inputs.size(); ++j) {
            const auto& input = training_inputs[j];
            const auto& expected_output = training_outputs[j];

            Eigen::VectorXf pred_output = feedfoward(input);

            total_epoch_error += Loss::mean_squared_error(pred_output, expected_output);

            backpropagate(input, expected_output);
            
            update_weights(learning_rate);
        }

        if ((i + 1) % 100 == 0) {
            std::cout << "Epoch " << i + 1 << "/" << epochs
                      << ", Average Error: " << total_epoch_error / training_inputs.size()
                      << std::endl;
        }
    }
}

