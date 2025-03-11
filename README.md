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
├── Components_tests/        # Tests for individual components of the networks
│   ├── test_activation.py   # Tests for activation functions
│   ├── test_loss.py         # Tests for loss functions
│   └── ...
│
├── ExampleDatasets/         # Example datasets in .mat format
│   ├── dataset1.mat
│   ├── dataset2.mat
│   └── ...
│
├── ResNet_tests/            # Tests specific to the ResNet implementation
│   ├── test_resnet_forward.py
│   ├── test_resnet_backward.py
│   └── ...
│
├── Standard_NN_tests/       # Tests specific to the standard neural network
│   ├── test_standard_nn_forward.py
│   ├── test_standard_nn_backward.py
│   └── ...
│
├── Whole_Network_tests/     # Tests involving the entire network architectures
│   ├── test_training_loop.py
│   ├── test_inference.py
│   └── ...
│
├── ResNet.py                # ResNet-style fully connected network implementation
├── Standard_NN.py           # Standard fully connected network implementation
└── README.md                # This file
```

## Training and Testing
The networks have been trained on example datasets stored in the `ExampleDatasets/` folder. Training is performed using **stochastic gradient descent (SGD)** with backpropagation. The models achieved **around 95% accuracy** on the provided datasets.

## Running the Code
To train the networks, run the respective scripts:
```bash
python Standard_NN.py
```
or
```bash
python ResNet.py
```
Ensure that the datasets are available in the `ExampleDatasets/` folder.

## Dependencies
The only dependency required to run the code is:
```bash
pip install numpy scipy
```
(No deep learning frameworks are used.)

## License
This project is released under the MIT License.

