# Neural Network from Scratch in C++

> An educational project to build an artificial neural network in C++, without relying on high-level frameworks. Aiming for a deep understanding of core machine learning algorithms like Feedforward and Backpropagation.

## 🛠️ Built With

-   **C++17:** The core language for the project.
-   **Eigen 3.4.0:** A C++ template library for linear algebra (matrices, vectors, etc.).
-   **Make:** The build automation system.

## 🚀 Getting Started

### Installation & Compilation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AraujoStreicher/neural-network-from-scratch-cpp.git
    cd neural-network-from-scratch-cpp
    ```

2.  **Compile the project:**
    Run `make` in the root folder. It will handle everything.
    ```bash
    make
    ```

4.  **Run the program:**
    After compilation, an executable named `neural_network` will be created.
    ```bash
    ./my_program
    ```

## ⚙️ Project Structure
```text
neural-network-from-scratch-cpp/
├── include/              # Header files (.hpp)
│   ├── Activation.hpp
│   ├── Layer.hpp
│   ├── Loss.hpp
│   └── NeuralNetwork.hpp
├── src/                  # Source code files (.cpp)
│   ├── Layer.cpp
│   └── NeuralNetwork.cpp
├── lib/                  # Third-party libraries
│   └── eigen-3.4.0/      
├── .gitignore            
├── main.cpp              
├── Makefile              
└── README.md   
```

## 💡 Usage Example: Solving XOR

The main program (`main.cpp`) is configured to train the network to solve the XOR problem. When you run it, you will see the training progress, with the error decreasing over epochs, followed by the final test.