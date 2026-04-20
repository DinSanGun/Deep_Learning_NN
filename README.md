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
│   └── assets/              # Report figures/assets (full report planned here)
├── Standard_NN.py           # Root compatibility wrapper
├── ResNet.py                # Root compatibility wrapper
├── requirements.txt
└── README.md
```

## Results

On the included example datasets, the project has previously achieved around **95% accuracy** (as documented in the original project notes).

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

A full project report and related assets will be maintained under `reports/`.

## License

This project is released under the MIT License.

