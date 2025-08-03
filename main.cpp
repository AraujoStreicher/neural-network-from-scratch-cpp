#include <iostream>
#include <cstdlib> 
#include <ctime> 
#include "lib/eigen-3.4.0/Eigen/Dense" 
#include "include/Layer.hpp" 
#include "include/NeuralNetwork.hpp"

int main() {
    srand(time(NULL)); 

    std::cout << "--- Testando a Classe NeuralNetwork ---" << std::endl;


    NeuralNetwork network;

    // 2 -> 3 -> 1
    Layer l1(2, 3);
    Layer l2(3, 1);
    network.addLayer(l1); // 1st layer: 3 neurons
    network.addLayer(l2); // 2nd layer: 1 neuron


    Eigen::VectorXf input(2);
    input << 0.8, 0.4;

    std::cout << "\nEntrada da Rede:\n" << input << std::endl;

    // Faz a previsão
    Eigen::VectorXf output = network.feedfoward(input);

    std::cout << "\nSaída Final da Rede:\n" << output << std::endl;

    return 0;
}