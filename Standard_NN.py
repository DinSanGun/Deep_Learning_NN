import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import loadmat


def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(z):
    """
    Compute the softmax of each column of the matrix Z with numerical stability.
    Z: Input matrix shaped: (classes x samples)
    Return: a softmax probability distribution for each sample (column)
    """
    max_value = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - max_value)  # For numerical stability
    exp_sum = np.sum(exp_z, axis=0, keepdims=True)
    return exp_z / exp_sum

def cross_entropy_loss(Y_pred, Y_true):
    """
    Computes the cross-entropy loss.

    Y_pred: Predicted probabilities (after softmax) in the shape (classes_num x samples_num)
    Y_true: True labels in the shape (classes_num x samples_num)

    Return: softmax cross-entropy loss (scalar value)
    """
    epsilon = 1e-15
    Y_pred_normalized = np.clip(Y_pred, epsilon, 1 - epsilon)  # Prevent log numerical issues
    loss = - np.sum(Y_true * np.log(Y_pred_normalized)) / Y_pred.shape[1] # Average on samples
    return loss

# Initialize weights and biases for the whole network
def initialize_network(input_dim, hidden_dims, output_dim):

    layers = []

    params_counter = 0

    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        W = np.random.randn(hidden_dim, prev_dim) * np.sqrt(2 / prev_dim) # He initialization
        b = np.zeros((hidden_dim, 1))
        layers.append((W, b))

        params_counter += hidden_dim * (prev_dim + 1)
        prev_dim = hidden_dim

    W_out = np.random.randn(output_dim, prev_dim) * np.sqrt(2 / prev_dim) # Initialize with small values
    b_out = np.zeros((output_dim, 1))

    params_counter += output_dim * (prev_dim + 1)
    print(f"Number of parameters: {params_counter}")

    layers.append((W_out, b_out))

    return layers

# Compute the forward pass for a multi-hidden-layer network
def forward_pass(X, layers):

    activations = [X]  # X has shape (features_num x samples_num)
    pre_activations_Z = []
    
    for i in range(len(layers) - 1):
        W, b = layers[i]
        Z = W @ activations[i] + b  # Pre-activation
        pre_activations_Z.append(Z)
        A = ReLU(Z)  # Activation
        activations.append(A)

    W_out, b_out = layers[-1]
    Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
    pre_activations_Z.append(Z_out)
    activations.append(Z_out) # The actual activation of output layer would be softmax

    return activations, pre_activations_Z

# Compute gradients for L layers network
def backpropagation(X, Y, activations, pre_activations, layers):
    """
    Params:
    X: Input data matrix - shape (input_dim x samples_num)
    Y: True labels matrix - shape (output_dim x samples_num)
    activations: List of activations (A) from forward pass
    pre_activations: List of pre-activations (Z) from forward pass
    layers: List of network layers (each layer contains weights and biases)

    Return:
    grads: List of gradients for weights and biases for all layers
    """
    samples_num = X.shape[1]  # Number of samples
    L = len(layers)  # Total number of layers
    grads = []

    # Output layer gradients
    Z_out = pre_activations[-1]
    Y_hat = softmax(Z_out)
    dZ = (Y_hat - Y) / samples_num  # Gradient of loss w.r.t. Z_out (output layer pre-activation)

    # Derivatives are calculated from last layer (output) to the beginning (input)
    for l in reversed(range(L)):

        W, b = layers[l]

        # Compute gradients for weights and biases
        dW = dZ @ activations[l].T 
        db = np.sum(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ      # Gradient w.r.t input to layer

        grads.insert(0, (dW, db)) # inserts in head of grads list

        if l > 0:  # Compute dZ for next layer
            Z = pre_activations[l - 1]
            dZ = dA * ReLU_derivative(Z)  # Derivative of RELU activation

    return grads

# Train the network using SGD with validation and accuracy computation
def train_network_sgd(X, Y, X_val, Y_val, input_dim, hidden_dims, output_dim, learning_rate=0.01, epochs=10, batch_size=32, accuracy_sample_size=1000):

    # Initialize network's parameters 
    layers = initialize_network(input_dim, hidden_dims, output_dim)  # (hidden_dims is an array of sizes)

    num_samples = X.shape[1]  # Number of training samples
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):

        # Shuffle training data
        indexes = np.random.permutation(num_samples)
        X = X[:, indexes]
        Y = Y[:, indexes]

        for start in range(0, num_samples, batch_size):

            end = start + batch_size
            X_batch = X[:, start:end]
            Y_batch = Y[:, start:end]

            # Forward pass
            activations, pre_activations_Z = forward_pass(X_batch, layers)
            Y_pred = softmax(pre_activations_Z[-1])

            # Compute loss
            loss = cross_entropy_loss(Y_pred, Y_batch)

            # Backpropagation
            grads = backpropagation(X_batch, Y_batch, activations, pre_activations_Z, layers)

            # Update weights and biases
            for i in range(len(layers)):
                W, b = layers[i]
                dW, db = grads[i]
                layers[i] = (W - learning_rate * dW, b - learning_rate * db)

        # Compute training loss for the epoch
        activations, pre_activations_Z = forward_pass(X, layers)
        Y_pred = softmax(pre_activations_Z[-1])
        epoch_train_loss = cross_entropy_loss(Y_pred, Y)
        train_losses.append(epoch_train_loss)

        # Compute validation loss for the epoch
        activations_val, pre_activations_Z_val = forward_pass(X_val, layers)
        Y_pred_val = softmax(pre_activations_Z_val[-1])
        epoch_val_loss = cross_entropy_loss(Y_pred_val, Y_val)
        val_losses.append(epoch_val_loss)

        # Compute accuracy using a random subset
        train_accuracy = compute_accuracy(X, Y, layers, sample_size=accuracy_sample_size)
        val_accuracy = compute_accuracy(X_val, Y_val, layers, sample_size=accuracy_sample_size)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.3f}, Validation Loss: {epoch_val_loss:.3f}, Train Acc: {train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%")

        # if(train_accuracy >= 0.9):
        #     learning_rate = learning_rate * 0.7

    # Plot accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), [acc * 100 for acc in train_accuracies], label='Train Accuracy', color='blue')
    plt.plot(range(epochs), [acc * 100 for acc in val_accuracies], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')  # Change the label to indicate percentage
    plt.title('Standard NN - GMM Dataset - 493 parameters')
    plt.legend()
    plt.grid(True)

    # Format the y-axis to show percentage values
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    plt.show()

def compute_accuracy(X, Y, layers, sample_size=1000):
    """
    Compute accuracy using a random subset of the data.
    """
    num_samples = X.shape[1]
    sample_indices = np.random.choice(num_samples, min(sample_size, num_samples), replace=False)
    X_sample = X[:, sample_indices]
    Y_sample = Y[:, sample_indices]

    # Forward pass
    activations, pre_activations_Z = forward_pass(X_sample, layers)
    Y_pred = softmax(pre_activations_Z[-1])

    # Convert predictions to class labels
    predicted_labels = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_sample, axis=0)

    # Compute accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

def main():
    """Main function to set up and train the network."""
    # Example data (to be replaced with actual dataset)
    np.random.seed(42)

    # Load data from the MATLAB file
    data = loadmat('../Data/GMMData.mat')
    X_training = data['Yt']  # Training input matrix
    Y_training = data['Ct']  # Training true labels matrix
    X_validation = data['Yv']    # Validation input matrix
    Y_validation = data['Cv']    # Validation true labels matrix

    # Neural network's dimensions
    input_features_n = X_training.shape[0]  # Number of features in input
    output_classes_n = Y_training.shape[0]  # Number of classes 

    input_dim = input_features_n
    hidden_dims = [8, 8, 8, 8, 8, 8]  
    output_dim = output_classes_n


    learning_rate = 0.00045
    epochs = 35
    batch_size = 8
    accuracy_sample_size = 500

    train_network_sgd(X_training, Y_training, X_validation, Y_validation, 
                      input_dim, hidden_dims, output_dim, 
                      learning_rate, epochs, batch_size, accuracy_sample_size)

    print("Training completed.")


# def main():
#     """Main function to set up and train the network."""
#     # Example data (to be replaced with actual dataset)
#     np.random.seed(42)

#     # Load data from the MATLAB file
#     data = loadmat('../Data/PeaksData.mat')
#     X_training = data['Yt']  # Training input matrix
#     Y_training = data['Ct']  # Training true labels matrix
#     X_validation = data['Yv']    # Validation input matrix
#     Y_validation = data['Cv']    # Validation true labels matrix

#     # Neural network's dimensions
#     input_features_n = X_training.shape[0]  # Number of features in input
#     output_classes_n = Y_training.shape[0]  # Number of classes 

#     input_dim = input_features_n
#     hidden_dims = [16, 8, 8, 8, 8]  
#     output_dim = output_classes_n


#     learning_rate = 0.0006
#     epochs = 40
#     batch_size = 8
#     accuracy_sample_size = 500

#     train_network_sgd(X_training, Y_training, X_validation, Y_validation, 
#                       input_dim, hidden_dims, output_dim, 
#                       learning_rate, epochs, batch_size, accuracy_sample_size)

#     print("Training completed.")



if __name__ == "__main__":
    main()




# ================================ TESTS =====================================
# === Below are all the functions that were used only for testing purposes === 

# # Initializes weights and biases from a given vector
# def initialize_by_vector(vector, N):
#     """
#     Params:
#     vector: 1D array of weights and biases for the entire network
#     N: number of neurons in any layer of the network 

#     Return:
#     layers: List of tuples (W, b), where W is the weight matrix and b is the bias vector of each layer
#     """
#     layers = []
#     slice_size = N * N + N
#     num_layers = len(vector) // slice_size
#     index = 0

#     for _ in range(num_layers):
#         W = vector[index:index + N * N].reshape(N, N)
#         index += N * N
#         b = vector[index:index + N].reshape(N, 1)
#         index += N
#         layers.append((W, b))

#     return layers

# # Flatten the gradients for all layers into a single 1D vector
# def flatten_gradients(grads):
#     """
#     Params:
#     grads: List of tuples (dW, db) representing gradients for the weights and biases of all the layers

#     Return:
#     flat_vector: A single 1-D array containing all the gradients of the network sequentially
#     """
#     flat_vector = []
#     for i in range(len(grads)):
#         dW, db = grads[i]
#         flat_vector.append(dW.flatten())  # Flatten weights gradient matrix
#         flat_vector.append(db.flatten())  # Flatten bias gradient vector
#     return np.concatenate(flat_vector)  # Combine gradients into one vector

# def whole_network_test(X, Y, vec, N):

#     # Initialization
#     layers = initialize_by_vector(vec, N)

#     # Forward pass
#     activations, pre_activations = forward_pass(X, layers)

#     # Network's output
#     Y_pred = softmax(pre_activations[-1])

#     # Loss calculation
#     loss = cross_entropy_loss(Y_pred, Y)

#     return loss, activations, pre_activations, layers

# def whole_network_gradient(X, Y, activations, pre_activations, layers):

#     grads = backpropagation(X, Y, activations, pre_activations, layers)
#     return flatten_gradients(grads)

# def forward_pass_i(X, layers, i):
#     """
#     Params:
#     X: Input matrix of shape (features_num x samples_num)
#     layers: List of all network's layers, each containing weights and biases
#     i: Index of layer to begin forward pass with

#     Return:
#     activations: List of activations beginning from layer i to the output layer
#     pre_activations_Z: List of pre-activations starting from layer i to the output layer
#     """

#     # Initializing layers before i (so we can use backpropagation as it is - does not affect the test)
#     activations = [np.random.randn(*layers[j][0].T.shape) for j in range(i)]
#     pre_activations_Z = [np.random.randn(*layers[j][0].shape) for j in range(i)]

#     # Start from layer i
#     activations.append(X)
#     for j in range(i, len(layers) - 1):
#         W, b = layers[j]
#         Z = W @ activations[-1] + b  # Pre-activation
#         pre_activations_Z.append(Z)
#         A = np.RELU(Z)  # Activation
#         activations.append(A)

#     # Output layer
#     W_out, b_out = layers[-1]
#     Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
#     pre_activations_Z.append(Z_out)
#     activations.append(Z_out)

#     return activations, pre_activations_Z

# def jacobian_test_layer_i(X, W, b, Y, i, layers):

#     # Overwrite initialization in order to give a certain input for testing purposes
#     layers[i] = (W,b)

#     # Forward pass
#     activations, pre_activations = forward_pass_i(X, layers, i)

#     Y_hats = softmax(activations[-1])  # Softmax probabilities
#     epsilon = 1e-15
#     Y_hats_normalized = np.clip(Y_hats, epsilon, 1 - epsilon) # Prevent log issues

#     # Compute loss
#     loss = cross_entropy_loss(Y_hats_normalized, Y)

#     return loss, activations, pre_activations


# def main():
#     """Main function to set up and train the network."""
#     # Example data (to be replaced with actual dataset)
#     np.random.seed(42)

#     # Load data from the MATLAB file
#     data = loadmat('../digits/emnist-digits.mat')

#     # Extract the train struct
#     train_struct = data['dataset']['train'][0, 0]  # Accessing the train struct
#     test_struct  = data['dataset']['test'][0, 0]  # Accessing the train struct

#     X_training = train_struct['images'].T  # Extracting images field
#     Y_training = train_struct['labels'].T  # Extracting labels field

#     X_validation = test_struct['images'].T    # Validation input matrix
#     Y_validation = test_struct['labels'].T    # Validation true labels matrix

#     # Neural network's dimensions
#     input_features_n = X_training.shape[0]  # Number of features in input
#     output_classes_n = Y_training.shape[0]  # Number of classes 

#     input_dim = input_features_n
#     hidden_dims = [128, 128, 64, 64, 32]  
#     output_dim = output_classes_n


#     learning_rate = 0.0006
#     epochs = 40
#     batch_size = 8
#     accuracy_sample_size = 500

#     train_network_sgd(X_training, Y_training, X_validation, Y_validation, 
#                       input_dim, hidden_dims, output_dim, 
#                       learning_rate, epochs, batch_size, accuracy_sample_size)

#     print("Training completed.")