#include <iostream>
#include <cstdlib> 
#include <ctime> 
#include "lib/eigen-3.4.0/Eigen/Dense" 
#include "include/Layer.hpp" 
#include "include/NeuralNetwork.hpp"

int main() {
    srand(time(NULL));

    std::vector<Eigen::VectorXf> training_inputs;
    std::vector<Eigen::VectorXf> training_outputs;

    // data 0 XOR 0 = 0
    Eigen::VectorXf input1(2);
    input1 << 0, 0;
    training_inputs.push_back(input1);
    Eigen::VectorXf output1(1);
    output1 << 0;
    training_outputs.push_back(output1);

    // data 0 XOR 1 = 1
    Eigen::VectorXf input2(2);
    input2 << 0, 1;
    training_inputs.push_back(input2);
    Eigen::VectorXf output2(1);
    output2 << 1;
    training_outputs.push_back(output2);

    // data 1 XOR 0 = 1
    Eigen::VectorXf input3(2);
    input3 << 1, 0;
    training_inputs.push_back(input3);
    Eigen::VectorXf output3(1);
    output3 << 1;
    training_outputs.push_back(output3);

    // data 1 XOR 1 = 0
    Eigen::VectorXf input4(2);
    input4 << 1, 1;
    training_inputs.push_back(input4);
    Eigen::VectorXf output4(1);
    output4 << 0;
    training_outputs.push_back(output4);


    //2 -> 3 -> 1
    NeuralNetwork network;    
    Layer l1(2, 3);
    Layer l2(3, 1);
    network.addLayer(l1);
    network.addLayer(l2);


    int EPOCHS = 10000;
    double LEARNING_RATE = 0.1;
    network.train(training_inputs, training_outputs, EPOCHS, LEARNING_RATE);


    std::cout << "\n--- TESTS ---" << std::endl;

    for (const auto& test_input : training_inputs) {
        Eigen::VectorXf prediction = network.feedfoward(test_input);
        std::cout << "Input: [" << test_input(0) << ", " << test_input(1) << "] "
                  << "-> Predicted output: " << prediction(0) << std::endl;
    }

    return 0;
}