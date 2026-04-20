# Fully Connected Deep Learning Networks from Scratch

This repository contains implementations of two fully connected deep learning neural networks for classification:
- A **standard fully connected neural network**
- A **ResNet-inspired version**

Both networks are implemented **from scratch in Python**, using only **NumPy** (without external deep learning libraries like PyTorch or TensorFlow). The networks have been tested on example datasets and achieved **around 95% accuracy**.

## Features
- Fully connected architecture (no convolutional layers)
- Implemented using only NumPy
- Backpropagation and gradient computation verified through tests
- Softmax activation with cross-entropy loss for classification
- **SGD (Stochastic Gradient Descent) optimizer** for training
- **ReLU activation function** in hidden layers

## Repository Structure
```
├── src/                     # Core model/source code
│   ├── standard_nn.py
│   ├── resnet.py
│   └── classifier_functions.py
├── tests/                   # Gradient and whole-network verification scripts
│   ├── gradients/
│   └── integration/
├── experiments/             # Training/demo scripts and exploratory runs
├── data/example_datasets/   # .mat datasets used by scripts
├── reports/assets/          # Report/figure assets
├── Standard_NN.py           # Root compatibility wrapper (standard NN)
├── ResNet.py                # Root compatibility wrapper (ResNet)
└── README.md
```

## Training and Testing
The networks have been trained on example datasets stored in `data/example_datasets/`. Training is performed using **stochastic gradient descent (SGD)** with backpropagation. The models achieved **around 95% accuracy** on the provided datasets.

## Running the Code
To train the networks, run the respective scripts:
```bash
python Standard_NN.py
```
or
```bash
python ResNet.py
```
Ensure that the datasets are available in `data/example_datasets/`.

## Dependencies
The only dependency required to run the code is:
```bash
pip install numpy scipy
```
(No deep learning frameworks are used.)

## License
This project is released under the MIT License.

