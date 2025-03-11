import numpy as np
import matplotlib.pyplot as plt
from Part_1.classifier_functions import *

def label_generator(classes, samples):
    # Create an NxN matrix of zeros
    matrix = np.zeros((classes, samples), dtype=int)

    # For each row, set a random position to 1
    for i in range(samples):
        random_index = np.random.randint(0, classes)  # Choose a random index in the column
        matrix[random_index, i] = 1  # Set that position to 1

    return matrix

# Define the function F and its gradient g_F
F = lambda Y_pred, Y_true: cross_entropy_loss(Y_pred, Y_true)
g_F = lambda X, Y_true, Y_pred: compute_gradients(X, Y_true, Y_pred)

# Parameters
n_samples = 50
input_features = 20
output_classes = 10

Y = label_generator(output_classes, n_samples)
b = np.random.randn(output_classes)
X = np.random.randn(input_features, n_samples)
W = np.random.randn(output_classes, input_features)

d = np.random.randn(output_classes, input_features)
epsilon = 0.1

Y_pred = softmax( W @ X + b.reshape(-1,1) )

F0 = F(Y_pred, Y)
g_W, g_b = g_F(X, Y, Y_pred)

y0 = np.zeros(8)
y1 = np.zeros(8)

weights_gradient_times_d = np.sum(g_W * d)

print("k\terror order 1 \t\t error order 2")

# Main loop
for k in range(1, 9):
    epsk = epsilon * (0.5 ** k)
    Y_pred = softmax( (W + epsk * d) @ X + b.reshape(-1,1) )

    Fk = F(Y_pred, Y)
    F1 = F0 + epsk * weights_gradient_times_d 
    y0[k - 1] = abs(Fk - F0)
    y1[k - 1] = abs(Fk - F1)
    print(f"{k}\t{abs(Fk - F0):.6e}\t{abs(Fk - F1):.6e}")

# Plotting
plt.semilogy(range(1, 9), y0, label="Zero order approx")
plt.semilogy(range(1, 9), y1, label="First order approx")
plt.legend()
plt.title("Successful Weights Grad test in semilogarithmic plot")
plt.xlabel("k")
plt.ylabel("error")
plt.show()
