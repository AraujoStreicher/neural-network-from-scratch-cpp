#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include "Eigen/Dense"
#include "Layer.hpp"

class NeuralNetwork {
public:
    NeuralNetwork();
    
    void addLayer(Layer& layer);
    Eigen::VectorXf feedfoward(Eigen::VectorXf input);
    void backpropagate(const Eigen::VectorXf& input, const Eigen::VectorXf& actual_output);
    void update_weights(double learning_rate);
    void train( 
        const std::vector<Eigen::VectorXf>& training_inputs,\
        const std::vector<Eigen::VectorXf>& training_outputs,\
        int epochs,\
        double learning_rate
    );

    // pair for each layer : <grad_weights , grad_biases>
    std::vector<std::pair<Eigen::MatrixXf, Eigen::VectorXf>> gradients;

private:
    std::vector<Layer*> layers;
};


#endif