import numpy as np
import matplotlib.pyplot as plt
from ResNet import backpropagation_resnet, initialize_resnet, jacobian_tests_block_i

def label_generator(classes, samples):
    # Create an NxN matrix of zeros
    matrix = np.zeros((classes, samples), dtype=int)

    # For each row, set a random position to 1
    for i in range(samples):
        random_index = np.random.randint(0, classes)  # Choose a random index in the column
        matrix[random_index, i] = 1  # Set that position to 1

    return matrix

# Define the function F and its gradient g_F
F = lambda X, W, b, Y, i, blocks: jacobian_tests_block_i(X, W, b, Y, i, blocks)
g_F = lambda X, Y, activations, pre_activations, blocks: backpropagation_resnet(X, Y, activations, pre_activations, blocks)

# Parameters
n_samples = 8
input_features = 8
output_classes = 8

i = 0
j = 2

Y = label_generator(output_classes, n_samples)
b = np.random.randn(output_classes)
b = b.reshape(-1, 1)
X = np.random.randn(input_features, n_samples)
W = np.random.uniform(0, 1, size=(output_classes, input_features))

d_w = np.random.uniform(0, 1, size=(output_classes, input_features)) * 10
d_x = np.random.randn(input_features, n_samples)
d_b = np.random.randn(output_classes)
d_b = d_b.reshape(-1, 1)

epsilon = 0.2
layers = initialize_resnet(X.shape[0], [X.shape[0], X.shape[0]], Y.shape[0])

F0, activations, pre_activations = F(X, W, b, Y, i, layers)
grads = g_F(X, Y, activations, pre_activations, layers)

x0 = np.zeros(8)
x1 = np.zeros(8)

y0 = np.zeros(8)
y1 = np.zeros(8)

z0 = np.zeros(8)
z1 = np.zeros(8)

g_W, g_b, _ = grads[i][j]
_, __, g_X = grads[i][0]

g_b = g_b.T[0] # reshape g_b for np.dot

weights_jacobian_times_d = np.sum(g_W * d_w)
bias_gradient_times_d = np.dot(g_b, d_b.flatten())
input_gradient_times_d = np.sum(g_X * d_x)

print("k\terror order 1 \t\t error order 2")

# Main loop
for k in range(1, 9):
    epsk = epsilon * (0.5 ** k)

    Fk_w, _, __ = F(X, W + epsk * d_w, b, Y, i, layers)
    Fk_b, _, __ = F(X, W, b + epsk * d_b, Y, i, layers)
    Fk_x, _, __ = F(X + epsk * d_x, W, b, Y, i, layers)

    F1_w = F0 + epsk * weights_jacobian_times_d 
    F1_b = F0 + epsk * bias_gradient_times_d 
    F1_x = F0 + epsk * input_gradient_times_d

    x0[k - 1] = abs(Fk_b - F0)
    x1[k - 1] = abs(Fk_b - F1_b)

    y0[k - 1] = abs(Fk_w - F0)
    y1[k - 1] = abs(Fk_w - F1_w)

    z0[k - 1] = abs(Fk_x - F0)
    z1[k - 1] = abs(Fk_x - F1_x)


    print(f"{k}\t{abs(Fk_w - F0):.6e}\t{abs(Fk_w - F1_w):.6e}  |  {k}\t{abs(Fk_x - F0):.6e}\t{abs(Fk_x - F1_x):.6e} | {k}\t{abs(Fk_b - F0):.6e}\t{abs(Fk_b - F1_b):.6e} ")

# Plotting
plt.figure()
plt.semilogy(range(1, 9), y0, label="Zero order approx")
plt.semilogy(range(1, 9), y1, label="First order approx")
plt.legend()
plt.title(f"Weights Jacobian test in semilogarithmic plot - Block {i + 1} - Layer {j + 1}")
plt.xlabel("k")
plt.ylabel("error")
plt.show()

# Plotting
plt.figure()
plt.semilogy(range(1, 9), z0, label="Zero order approx")
plt.semilogy(range(1, 9), z1, label="First order approx")
plt.legend()
plt.title(f"Input Jacobian test in semilogarithmic plot - Block {i + 1}")
plt.xlabel("k")
plt.ylabel("error")
plt.show()

# Plotting
plt.figure()
plt.semilogy(range(1, 9), x0, label="Zero order approx")
plt.semilogy(range(1, 9), x1, label="First order approx")
plt.legend()
plt.title(f"Bias Grad test in semilogarithmic plot - Block {i + 1} - Layer {j + 1}")
plt.xlabel("k")
plt.ylabel("error")
plt.show()