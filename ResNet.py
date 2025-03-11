import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

# Provided code (unchanged)
def softmax(z):
    """
    Compute the softmax of each column of the matrix Z with numerical stability.
    Z: Input matrix of a form: classes x samples
    Returns a softmax probability distribution for each column (sample).
    """
    max_value = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - max_value)  # Prevent numerical issues
    exp_sum = np.sum(exp_z, axis=0, keepdims=True)
    return exp_z / exp_sum

def cross_entropy_loss(Y_pred, Y_true):
    """
    Computes the cross-entropy loss.

    Y_pred: Predicted probabilities (after softmax), shape (classes_num, samples_num)
    Y_true: True labels, shape (classes_num, samples_num)

    Return:
    loss: The softmax cross-entropy loss (scalar)
    """
    epsilon = 1e-15
    Y_pred_clipped = np.clip(Y_pred, epsilon, 1 - epsilon)  # Prevent log issues
    loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / Y_pred.shape[1]
    return loss

# Initialize weights and biases for a ResNet with 3-layer blocks
def initialize_resnet(input_dim, hidden_dims, output_dim):

    blocks = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        
        block = [] # Block of 3 layers

        for _ in range(3):
            W = np.random.randn(hidden_dim, prev_dim) * np.sqrt(2 / prev_dim)
            b = np.zeros((hidden_dim, 1))
            block.append((W, b))
            prev_dim = hidden_dim
        blocks.append(block)
    
    # Output layer
    W_out = np.random.randn(output_dim, prev_dim) * np.sqrt(2 / prev_dim)
    b_out = np.zeros((output_dim, 1))
    blocks.append([(W_out, b_out)])  # Output is considered a single block
    return blocks

# Compute the forward pass for a ResNet
def forward_pass_resnet(X, blocks):

    activations = [X] 
    pre_activations = []

    for block_index in range(len(blocks) - 1):

        block = blocks[block_index]
        residual = activations[-1]  # Save the input as residual

        for layer_index in range(len(block)):
            W, b = block[layer_index]
            Z = W @ activations[-1] + b  # Pre-activation
            pre_activations.append(Z)
            A = ReLU(Z)  # Activation
            activations.append(A)

        if residual.shape != activations[-1].shape: # Match dimensions of residual to activation of the block
            residual = match_shape(residual, activations[-1].shape) 

        # Add skip connection output to final layer's activation in the block)
        activations[-1] += residual

    # Output layer
    W_out, b_out = blocks[-1][0]
    Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
    pre_activations.append(Z_out)
    activations.append(Z_out)

    return activations, pre_activations

def match_shape(residual, target_shape): # Linear transformation
    modified_W = np.random.randn(target_shape[0], residual.shape[0]) * 0.01  # Small random weights
    modified_B = np.zeros((target_shape[0], 1))  # Bias initialization
    return modified_W @ residual + modified_B  # Equivalent to 1x1 conv


# Compute gradients for a ResNet with blocks of 3 layers
def backpropagation_resnet(X, Y, activations, pre_activations, blocks):
    """
    Params:
    X: Input data matrix - shape (input_dim x samples_num)
    Y: True labels - shape (output_dim x samples_num)
    activations: List of activations from forward pass
    pre_activations: List of pre-activations from forward pass
    blocks: List of blocks containing layers with weights and biases

    Return:
    grads: List of gradients for weights and biases for all the layers
    """
    samples_num = X.shape[1]  # Number of samples
    grads = []

    # Output layer gradients
    Z_out = pre_activations[-1]
    Y_hat = softmax(Z_out)
    dZ = (Y_hat - Y) / samples_num  # Gradient of loss w.r.t. Z_out (output layer pre-activation)

    # Gradients for output layer
    W_out, b_out = blocks[-1][0]
    dW_out = dZ @ activations[-2].T
    db_out = np.sum(dZ, axis=1, keepdims=True)
    dA = W_out.T @ dZ
    grads.append([(dW_out, db_out)])

    # Backpropagate through blocks
    pre_activation_index = len(pre_activations) - 2
    activation_index = len(activations) - 3

    for block in reversed(blocks[:-1]):

        residual = dA  # Store residual gradient for skip connection
        block_grads = []

        for layer in reversed(block):

            W, b = layer
            Z = pre_activations[pre_activation_index]
            A_prev = activations[activation_index]

            dZ = dA * ReLU_derivative(Z)  # Gradient of loss w.r.t. Z
            dW = dZ @ A_prev.T
            db = np.sum(dZ, axis=1, keepdims=True)

            dA = W.T @ dZ

            block_grads.insert(0, (dW, db))

            # Update indices
            pre_activation_index -= 1
            activation_index -= 1

        # Add skip connection gradient

        if residual.shape != dA.shape: # Match dimensions of residual gradient
            residual = match_shape(residual, dA.shape) 

        dA += residual
        grads.insert(0, block_grads)

    return grads

# Train ResNet using SGD
def train_resnet_sgd(X, Y, X_val, Y_val, input_dim, hidden_dims, output_dim, learning_rate=0.01, epochs=10, batch_size=32, accuracy_sample_size=1000):

    # Initialize network's parameters 
    blocks = initialize_resnet(input_dim, hidden_dims, output_dim)  # (hidden_dims is an array of sizes)

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
            activations, pre_activations_Z = forward_pass_resnet(X_batch, blocks)
            Y_pred = softmax(pre_activations_Z[-1])

            # Compute loss
            loss = cross_entropy_loss(Y_pred, Y_batch)

            # Backpropagation
            grads = backpropagation_resnet(X_batch, Y_batch, activations, pre_activations_Z, blocks)

            # Update weights and biases
            for block, block_grads in zip(blocks, grads):
                for i in range(len(block)):
                    W, b = block[i]
                    dW, db = block_grads[i]
                    block[i] = (W - learning_rate * dW, b - learning_rate * db)

        # Compute training loss for the epoch
        activations, pre_activations_Z = forward_pass_resnet(X, blocks)
        Y_pred = softmax(pre_activations_Z[-1])
        epoch_train_loss = cross_entropy_loss(Y_pred, Y)
        train_losses.append(epoch_train_loss)

        # Compute validation loss for the epoch
        activations_val, pre_activations_Z_val = forward_pass_resnet(X_val, blocks)
        Y_pred_val = softmax(pre_activations_Z_val[-1])
        epoch_val_loss = cross_entropy_loss(Y_pred_val, Y_val)
        val_losses.append(epoch_val_loss)

        # Compute accuracy using a random subset
        train_accuracy = compute_accuracy(X, Y, blocks, sample_size=accuracy_sample_size)
        val_accuracy = compute_accuracy(X_val, Y_val, blocks, sample_size=accuracy_sample_size)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.3f}, Validation Loss: {epoch_val_loss:.3f}, Train Acc: {train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%")

    # Plot accuracy graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), [acc * 100 for acc in train_accuracies], label='Train Accuracy', color='blue')
    plt.plot(range(epochs), [acc * 100 for acc in val_accuracies], label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')  # Change the label to indicate percentage
    plt.title('Accuracy vs Epochs - ResNet - Peaks Dataset - 453 parameters')
    plt.legend()
    plt.grid(True)

    # Format the y-axis to show percentage values
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    plt.show()

def compute_accuracy(X, Y, blocks, sample_size=1000):
    """
    Compute accuracy using a random subset of the data.
    """
    num_samples = X.shape[1]
    sample_indices = np.random.choice(num_samples, min(sample_size, num_samples), replace=False)
    X_sample = X[:, sample_indices]
    Y_sample = Y[:, sample_indices]

    # Forward pass
    activations, pre_activations_Z = forward_pass_resnet(X_sample, blocks)
    Y_pred = softmax(pre_activations_Z[-1])

    # Convert predictions to class labels
    predicted_labels = np.argmax(Y_pred, axis=0)
    true_labels = np.argmax(Y_sample, axis=0)

    # Compute accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

# ======================== MAIN =============================

def main():
    """Main function to set up and train the network."""
    # Example data (to be replaced with actual dataset)
    np.random.seed(17)

    # Load data from the MATLAB file
    data = loadmat('../Data/PeaksData.mat')
    X_training = data['Yt']  # Training input matrix
    Y_training = data['Ct']  # Training true labels matrix
    X_validation = data['Yv']    # Validation input matrix
    Y_validation = data['Cv']    # Validation true labels matrix

    # Neural network's dimensions
    input_features_n = X_training.shape[0]  # Number of features in input
    output_classes_n = Y_training.shape[0]  # Number of classes 

    input_dim = input_features_n
    hidden_dims = [8, 8]  # 5 hidden layers
    output_dim = output_classes_n


    learning_rate = 0.0004
    epochs = 35
    batch_size = 8
    accuracy_sample_size = 500

    train_resnet_sgd(X_training, Y_training, X_validation, Y_validation, 
                      input_dim, hidden_dims, output_dim, 
                      learning_rate, epochs, batch_size, accuracy_sample_size)

    print("Training completed.")

if __name__ == "__main__":
    main()

# ================================ TESTS =====================================
# === Below are all the functions that were used only for testing purposes === 


# # Perform the forward pass starting from block i.
# def forward_pass_resnet_i(X, blocks, i):
#     """
#     Params:
#     X: Input matrix of shape (features_num x samples_num)
#     blocks: List of network blocks, each containing layers with weights and biases
#     i: Index of the starting block

#     Return:
#     activations: List of activations starting from block i to the output layer.
#     pre_activations: List of pre-activations starting from block i to the output layer.
#     """
#     activations = [
#         np.random.randn(W.shape[1], X.shape[1]) for block in blocks[0 : i] for W, _ in block ]
    
#     pre_activations = [
#         np.random.randn(W.shape[0], X.shape[1]) for block in blocks[0 : i] for W, _ in block ]

#     activations.append(X)  # Start with the input at block i

#     for block_index in range(i, len(blocks) - 1):

#         block = blocks[block_index]
#         residual = activations[-1]  # Save input as residual

#         for W, b in block:
#             Z = W @ activations[-1] + b  # Pre-activation
#             pre_activations.append(Z)
#             A = np.tanh(Z)  # Activation
#             activations.append(A)

#         # Add skip connection output to final layer's activation
#         activations[-1] += residual

#     # Output layer
#     W_out, b_out = blocks[-1][0]
#     Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
#     pre_activations.append(Z_out)
#     activations.append(Z_out)

#     return activations, pre_activations

# def jacobian_tests_block_i(X, W, b, Y, i, blocks):

#     # Overwrite initialization in order to give a certain input for testing purposes
#     blocks[i][2] = (W,b) # change 3rd layer of block i

#     # Forward pass
#     activations, pre_activations = forward_pass_resnet_i(X, blocks, i)

#     Y_hats = softmax(activations[-1])  # Softmax probabilities
#     epsilon = 1e-15
#     Y_hats_normalized = np.clip(Y_hats, epsilon, 1 - epsilon) # Prevent log issues

#     # Compute loss
#     loss = cross_entropy_loss(Y_hats_normalized, Y)

#     return loss, activations, pre_activations

# # Initializes a ResNet from a 1-D vector
# def initialize_by_vector(vec, N):
#     """
#     Params:
#     vector: 1D array containing weights and biases for the entire network
#     N: Number of neurons in any hidden layer of the network

#     Returns:
#     blocks: List of residual blocks, each containing (W, b) tuples
#     """
#     blocks = []
#     slice_size = N * N + N
#     num_layers = len(vec) // slice_size
#     num_blocks = num_layers // 3  # Each block has 3 layers
#     index = 0
    
#     # Initialize residual blocks
#     for i in range(num_blocks):

#         block = []
#         for j in range(3):  # 3 layers

#             W = vec[index:index + N * N].reshape(N, N)
#             index += N * N

#             b = vec[index:index + N].reshape(N, 1)
#             index += N

#             block.append((W, b))

#         blocks.append(block)
    
#     # Initialize output layer
#     W_out = vec[index:index + N * N].reshape(N, N)
#     index += N * N
#     b_out = vec[index:index + N].reshape(N, 1)
    
#     blocks.append([(W_out, b_out)])  # Output layer is a single block
    
#     return blocks

# def whole_network_test(X, Y, vec, N):

#     # Initialization
#     blocks = initialize_by_vector(vec, N)

#     # Forward pass
#     activations, pre_activations = forward_pass_resnet(X, blocks)

#     # Network's output
#     Y_pred = softmax(pre_activations[-1])

#     # Loss calculation
#     loss = cross_entropy_loss(Y_pred, Y)

#     return loss, activations, pre_activations, blocks

# # Flatten the gradients for all the layers in the network into a single 1-D vector
# def flatten_gradients(grads):
#     """
#     Params:
#     grads: List of blocks, each containing tuples (dW, db) representing gradients

#     Return:
#     flat_vector: A single 1-D array containing all gradients sequentially
#     """

#     flatten_vector = []
#     for block in grads:
#         for dW, db, dA in block:
#             flatten_vector.append(dW.flatten())  # Flatten weights gradient matrix
#             flatten_vector.append(db.flatten())  # Flatten bias gradient vector
#     return np.concatenate(flatten_vector)  # Combine all into one vector

# def whole_network_gradient(X, Y, activations, pre_activations, blocks):

#     grads = backpropagation_resnet(X, Y, activations, pre_activations, blocks)
#     return flatten_gradients(grads)
