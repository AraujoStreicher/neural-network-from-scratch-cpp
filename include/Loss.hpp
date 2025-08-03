#ifndef LOSS_HPP
#define LOSS_HPP

#include "Eigen/Dense"

namespace Loss {

    inline double mean_squared_error(const Eigen::VectorXf& output_pred, const Eigen::VectorXf& output_real){
        Eigen::VectorXf diff = output_pred - output_real;
        
        return diff.squaredNorm() / diff.size();
    }

    inline Eigen::VectorXf mean_squared_error_derivative(const Eigen::VectorXf& output_pred, const Eigen::VectorXf& output_real){
        // 2 * (y_pred - y) / n
        return 2.0 * (output_pred - output_real) / output_pred.size();
    }
}

#endif