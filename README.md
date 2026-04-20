# Deep Learning Neural Networks From Scratch (Python)

This project implements deep learning classification models from scratch in Python, without PyTorch or TensorFlow.  
It includes both a standard fully connected network and a residual-style fully connected network, along with gradient/Jacobian verification scripts.

## Core Components

- Softmax regression output layer
- Cross-entropy loss
- SGD with mini-batch training
- Standard fully connected neural network
- Residual neural network (fully connected ResNet-style blocks)
- Gradient and Jacobian verification (component-level and whole-network)

## Repository Structure

```text
Deep_Learning_NN/
├── src/                     # Core source/model code
│   ├── standard_nn.py
│   ├── resnet.py
│   └── classifier_functions.py
├── tests/                   # Verification scripts
│   ├── gradients/           # Gradient/Jacobian checks
│   └── integration/         # Whole-network gradient checks
├── experiments/             # Training and exploratory scripts
├── data/
│   └── example_datasets/    # .mat datasets used by training scripts
├── reports/
│   ├── Deep Learning Project Report.pdf
│   └── assets/              # Report figures/assets
├── Standard_NN.py           # Root compatibility wrapper
├── ResNet.py                # Root compatibility wrapper
├── requirements.txt
└── README.md
```

## Results

From the project report, training on the included datasets reached:

- Around **92.5%** average accuracy on Peaks (best standard-network setting reported)
- Around **95%** average accuracy on GMM (standard network)
- Up to **96%** average accuracy on GMM (ResNet configuration)

Gradient/Jacobian verification scripts show the expected first-order and second-order error trends, supporting correctness of the implemented derivatives.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python Standard_NN.py
python ResNet.py
```

Run verification scripts:

```bash
python tests/gradients/grad_test_matrix_weights.py
python tests/gradients/grad_test_matrix_bias.py
python tests/gradients/standard_derivatives_test_v3.py
python tests/gradients/resnet_blocks_test.py
python tests/integration/whole_network_test.py
python tests/integration/resnet_whole_network_test.py
```

## Datasets

Datasets are stored in `data/example_datasets/` as `.mat` files.

## Report

The full write-up is available at:

- [`reports/Deep Learning Project Report.pdf`](reports/Deep%20Learning%20Project%20Report.pdf)

It documents:

- Derivation and implementation details for softmax, cross-entropy, and SGD
- Layer/block-level Jacobian and gradient verification
- Whole-network gradient checks
- Hyperparameter exploration and training outcomes on Peaks and GMM

## License

This project is released under the MIT License.

