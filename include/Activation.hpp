#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>
#include <cmath>

namespace Activation {

    // Sigmoid activation function
    inline Eigen::VectorXf sigmoid(const Eigen::VectorXf& z) {
        return 1.0 / (1.0 + (-z.array()).exp());
    }

    // Sigmoid derivative
    inline Eigen::VectorXf sigmoid_derivative(const Eigen:: VectorXf& z){
        Eigen::VectorXf sig_z = sigmoid(z);
        return sig_z.array() * (1.0 - sig_z.array());
    }

}

#endif