# Micrograd (Custom Build)
A minimal and educational implementation of a scalar-based automatic differentiation engine, inspired by Karpathy's Micrograd, written from scratch in Python using pure OOP principles.

This version includes:

Gradient computation using reverse-mode autodiff

Support for basic operations (+, -, *, /, **)

tanh and exp activations

Graph visualization using graphviz

A sample implementation of a single neuron

🔧 Features
Reverse-mode autodiff (backpropagation) over scalar computation graphs.

Operator overloading for intuitive mathematical expressions.

Graph tracing and visualization with graphviz.

Gradient checking via finite difference approximation.

Basic MLP neuron demo and expression tracing.

📁 Structure
Main Components
Value class: the core data structure representing a scalar value with autodiff support.

Overloaded arithmetic operators: +, -, *, /, **, tanh(), exp().

backward(): computes gradients via backpropagation.

trace() and draw_dot(): for visualization of computation graphs.

📈 Example Usage
python
Copy
Edit
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

L.backward()
draw_dot(L)
Output Graph
This will produce a computation graph showing forward values and backward gradients.

🧠 Neuron Example
python
Copy
Edit
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.881, label='b')

n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()

draw_dot(o)
📦 Dependencies
numpy

matplotlib

graphviz

Install dependencies with:

bash
Copy
Edit
pip install numpy matplotlib graphviz
For Graphviz rendering, ensure Graphviz is installed on your system:

On Ubuntu: sudo apt install graphviz

On Mac: brew install graphviz

On Windows: Install Graphviz

🧪 Gradient Checking
Finite difference approximations are used to validate gradients manually:

python
Copy
Edit
def lol():
    ...
    print((L2 - L1) / h)
🛠️ To Do
Add support for more activation functions (ReLU, sigmoid)

Extend to vectors and tensors

Implement basic MLP

🤝 Acknowledgments
Karpathy's Micrograd for the original idea and structure.

This version is a learning project with extended features and annotations.

