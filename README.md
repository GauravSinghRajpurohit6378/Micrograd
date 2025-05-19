# Micrograd (Custom Build)

This project implements a minimalist reverse-mode automatic differentiation (autodiff) engine, inspired by micrograd. It constructs a computational graph dynamically during forward operations and performs backpropagation to compute gradients with respect to inputs. Each scalar value is represented as a node in the graph, storing its computed value, gradient, and a reference to the operation and child nodes that produced it.

Key features include:

A custom Value class that supports arithmetic operations (+, -, *, /, **) and elementary functions like tanh and exp, all while tracking gradient flow.

Manual graph construction for educational clarity, allowing detailed insight into how autodiff frameworks work under the hood.

Support for visualizing the computational graph using Graphviz, aiding in understanding the chain of operations and dependencies.

Backpropagation implemented through a topologically sorted traversal of the graph, ensuring correct gradient computation even in complex expressions.

This project is intended as a learning tool for understanding the fundamentals of backpropagation and autodiff systems, such as those used in deep learning frameworks like PyTorch and TensorFlo

🤝 Acknowledgments
Karpathy's Micrograd for the original idea and structure.

This version is a learning project with extended features and annotations.

