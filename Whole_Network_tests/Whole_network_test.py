import numpy as np
import matplotlib.pyplot as plt
from Standard_NN_L_layers import whole_network_gradient, whole_network_test

def label_generator(classes, samples):
    # Create an NxN matrix of zeros
    matrix = np.zeros((classes, samples), dtype=int)

    # For each row, set a random position to 1
    for i in range(samples):
        random_index = np.random.randint(0, classes)  # Choose a random index in the column
        matrix[random_index, i] = 1  # Set that position to 1

    return matrix

# Define the function F and its gradient g_F
F = lambda X, Y, vec, features_num: whole_network_test(X, Y, vec, features_num)
g_F = lambda X, Y, activations, pre_activations, layers: whole_network_gradient(X, Y, activations, pre_activations, layers)

# Parameters
n_samples = 4
input_features = 4
output_classes = 4
L = 3
layer_weights_and_bias_size = ((input_features * input_features) + input_features)

Y = label_generator(output_classes, n_samples)
X = np.random.randn(input_features, n_samples)

vec = np.random.rand(layer_weights_and_bias_size * L)
d_vec = np.random.rand(layer_weights_and_bias_size * L)

epsilon = 0.1

F0, activations, pre_activations, layers = F(X, Y, vec, input_features)
grad = g_F(X, Y, activations, pre_activations, layers)

y0 = np.zeros(8)
y1 = np.zeros(8)

gradient_times_d = np.dot(grad, d_vec)

print("k\terror order 1 \t\t error order 2")

# Main loop
for k in range(1, 9):
    epsk = epsilon * (0.5 ** k)

    Fk, _, __, ___ = F(X, Y, vec + epsk * d_vec, input_features)

    F1 = F0 + epsk * gradient_times_d 

    y0[k - 1] = abs(Fk - F0)
    y1[k - 1] = abs(Fk - F1)

    print(f"{k}\t{abs(Fk - F0):.6e}\t{abs(Fk - F1):.6e}")

# Plotting
plt.figure()
plt.semilogy(range(1, 9), y0, label="Zero order approx")
plt.semilogy(range(1, 9), y1, label="First order approx")
plt.legend()
plt.title(f"Standard NN Gradient test - All Parameters - semilogarithmic plot")
plt.xlabel("k")
plt.ylabel("error")
plt.show()