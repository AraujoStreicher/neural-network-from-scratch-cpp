#include "Layer.hpp"
#include "Activation.hpp" 

Layer::Layer(int inputSize, int outputSize) {
    // Initializes the weight matrix with random values within the range [-1, 1]
    // shape: (number of neurons, number of inputs)
    weights = Eigen::MatrixXf::Random(outputSize, inputSize); 

    // Initializes the bias vector with random values within the range [-1, 1]
    // shape: (number of neurons)
    biases = Eigen::VectorXf::Random(outputSize);
}
    


Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input) {
    // output: z = (weights * input) + biases
    Eigen::VectorXf z = (weights * input) + biases;

    return Activation::sigmoid(z); 
}