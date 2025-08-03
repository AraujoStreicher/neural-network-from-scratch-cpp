#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense> 

class Layer {
public:
    Eigen::VectorXf last_input;  
    Eigen::VectorXf last_z;      

    // Constructor: initializes weights and biases
    // inputSize: number of inputs to the layer
    // outputSize: number of neurons in the layer
    Layer(int inputSize, int outputSize);

    // forward: propagates the input through the layer
    // input: input vector of size inputSize
    Eigen::VectorXf forward(const Eigen::VectorXf& input);

private:
    Eigen::MatrixXf weights; // Weight matrix
    Eigen::VectorXf biases;  // Bias vector
};

#endif