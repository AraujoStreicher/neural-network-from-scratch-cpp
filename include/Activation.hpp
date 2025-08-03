#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>
#include <cmath>

namespace Activation {

    // Sigmoid activation function
    inline Eigen::VectorXf sigmoid(const Eigen::VectorXf& z) {
        return 1.0 / (1.0 + (-z.array()).exp());
    }

}

#endif