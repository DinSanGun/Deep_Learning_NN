import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import loadmat
from Part_1.classifier_functions import *

# Load data from the MATLAB file
data = loadmat('Data/PeaksData.mat')
X_training = data['Yt']  # Training input matrix
Y_training = data['Ct']  # Training true labels matrix
X_validation = data['Yv']    # Validation input matrix
Y_validation = data['Cv']    # Validation true labels matrix

# Neural network's dimensions
input_features_n = X_training.shape[0]  # Numbe of features
output_classes_n = Y_training.shape[0]  # Number of classes 
samples_n = X_training.shape[1]
validation_samples_n = X_validation.shape[1]

# Initialize weights and biases
W = np.random.rand(output_classes_n, input_features_n) - 0.5  # Output layer weights
b = np.zeros((output_classes_n, 1))  # Output layer biases

# Training parameters
num_epochs = 15
batch_size = 256
learning_rate = 0.01

# Arrays to store the accuracies after each epoch
train_accuracies = []
val_accuracies = []

# Stochastic Gradient Descent
for epoch in range(num_epochs):

    # Shuffle training data
    indexes = np.random.permutation(samples_n)

    # Reorder input and true labels matrices
    X_training = X_training[:, indexes]
    Y_training = Y_training[:, indexes]

    # Training phase 
    for start in range(0, samples_n, batch_size):

        end = start + batch_size

        # Select mini-batch
        X_batch = X_training[:,start:end]
        Y_batch = Y_training[:,start:end]

        # Forward pass 
        # Broadcast the vector b as a column vector along the columns of the product matrix
        z = W @ X_batch + b.reshape(-1,1) 
        Y_pred = softmax(z)

        # Compute loss
        loss = cross_entropy_loss(Y_pred, Y_batch)

        # Compute gradients
        grad_W, grad_b = compute_gradients(X_batch, Y_batch, Y_pred)

        # Update weights and biases
        W -= learning_rate * grad_W
        b -= learning_rate * grad_b.reshape(-1,1)

    # Random subset for accuracy evaluation
    subset_n = 500  # Subset size for evaluating success percentage
    train_indexes = np.random.choice(samples_n, subset_n, replace=False) # Choose without replacement
    validation_indexes = np.random.choice(validation_samples_n, subset_n, replace=False)

    # Training accuracy
    X_training_subset = X_training[:, train_indexes]
    Y_training_subset = Y_training[:, train_indexes]

    z_train = W @ X_training_subset + b.reshape(-1,1)
    Y_training_pred = softmax(z_train)

    # Identify which class the model predicts for each training sample
    train_predictions = np.argmax(Y_training_pred, axis=0)
    train_labels = np.argmax(Y_training_subset, axis=0)

    # Compare the predicted class labels with the true labels
    # Sums and averages the number of correct predictions the model has made
    train_accuracy = np.sum(train_predictions == train_labels) / subset_n

    # Validation accuracy
    X_validation_subset = X_validation[:, validation_indexes]
    Y_validation_subset = Y_validation[:, validation_indexes]

    z_val = W @ X_validation_subset + b.reshape(-1,1)
    Y_validation_pred = softmax(z_val)

    # Identify which class the model predicts for each training sample
    val_predictions = np.argmax(Y_validation_pred, axis=0)
    val_labels = np.argmax(Y_validation_subset, axis=0)

    # Compare the predicted class labels with the true labels
    val_accuracy = np.mean(val_predictions == val_labels)

    # Store accuracies
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, "
          f"Train Accuracy: {train_accuracy*100:.2f}%, Validation Accuracy: {val_accuracy*100:.2f}%")

# Plot accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), [acc * 100 for acc in train_accuracies], label='Train Accuracy', color='blue')
plt.plot(range(num_epochs), [acc * 100 for acc in val_accuracies], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')  # Change the label to indicate percentage
plt.title('Accuracy vs Epochs (Single Layer Network)')
plt.legend()
plt.grid(True)

# Format the y-axis to show percentage values
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

plt.show()
