import numpy as np

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
            W = np.random.randn(hidden_dim, prev_dim) * 0.01
            b = np.zeros((hidden_dim, 1))
            block.append((W, b))
            prev_dim = hidden_dim
        blocks.append(block)
    
    # Output layer
    W_out = np.random.randn(output_dim, prev_dim) * 0.01
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
            A = np.tanh(Z)  # Activation
            activations.append(A)

        # Add skip connection output to final layer's activation in the block
        activations[-1] += residual

    # Output layer
    W_out, b_out = blocks[-1][0]
    Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
    pre_activations.append(Z_out)
    activations.append(Z_out)

    return activations, pre_activations

# Perform the forward pass starting from block i.
def forward_pass_resnet_i(X, blocks, i):
    """
    Params:
    X: Input matrix of shape (features_num x samples_num)
    blocks: List of network blocks, each containing layers with weights and biases
    i: Index of the starting block

    Return:
    activations: List of activations starting from block i to the output layer.
    pre_activations: List of pre-activations starting from block i to the output layer.
    """
    activations = [
        np.random.randn(W.shape[1], X.shape[1]) for block in blocks[0 : i] for W, _ in block ]
    
    pre_activations = [
        np.random.randn(W.shape[0], X.shape[1]) for block in blocks[0 : i] for W, _ in block ]

    activations.append(X)  # Start with the input at block i

    for block_index in range(i, len(blocks) - 1):

        block = blocks[block_index]
        residual = activations[-1]  # Save input as residual

        for W, b in block:
            Z = W @ activations[-1] + b  # Pre-activation
            pre_activations.append(Z)
            A = np.tanh(Z)  # Activation
            activations.append(A)

        # Add skip connection output to final layer's activation
        activations[-1] += residual

    # Output layer
    W_out, b_out = blocks[-1][0]
    Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
    pre_activations.append(Z_out)
    activations.append(Z_out)

    return activations, pre_activations

def jacobian_tests_block_i(X, W, b, Y, i, blocks):

    # Overwrite initialization in order to give a certain input for testing purposes
    blocks[i][2] = (W,b) # change 3rd layer of block i

    # Forward pass
    activations, pre_activations = forward_pass_resnet_i(X, blocks, i)

    Y_hats = softmax(activations[-1])  # Softmax probabilities
    epsilon = 1e-15
    Y_hats_normalized = np.clip(Y_hats, epsilon, 1 - epsilon) # Prevent log issues

    # Compute loss
    loss = cross_entropy_loss(Y_hats_normalized, Y)

    return loss, activations, pre_activations

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
    grads.append([(dW_out, db_out, dA)])

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

            dZ = dA * (1 - np.tanh(Z) ** 2)  # Gradient of loss w.r.t. Z
            dW = dZ @ A_prev.T
            db = np.sum(dZ, axis=1, keepdims=True)

            dA = W.T @ dZ

            block_grads.insert(0, (dW, db, dA))

            # Update indices
            pre_activation_index -= 1
            activation_index -= 1

        # Add skip connection gradient
        dA += residual
        grads.insert(0, block_grads)

    return grads

# Train ResNet using SGD
def train_resnet_sgd(X, Y, input_dim, hidden_dims, output_dim, learning_rate=0.01, epochs=10, batch_size=32):

    # Initialize parameters
    blocks = initialize_resnet(input_dim, hidden_dims, output_dim)

    num_samples = X.shape[1]  # Num of samples
    losses = []

    for epoch in range(epochs):
        indexes = np.random.permutation(num_samples)
        X = X[:, indexes]  # Shuffle samples
        Y = Y[:, indexes]  # Shuffle labels

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch = X[:, start:end]  # Select batch
            Y_batch = Y[:, start:end]  # Select batch

            # Forward pass
            activations, pre_activations = forward_pass_resnet(X_batch, blocks)
            Y_pred = softmax(pre_activations[-1])  # Output of the network after softmax

            # Compute loss
            loss = cross_entropy_loss(Y_pred, Y_batch)

            # Backpropagation
            grads = backpropagation_resnet(X_batch, Y_batch, activations, pre_activations, blocks)

            # Update weights and biases
            for block, block_grads in zip(blocks, grads):
                for i in range(len(block)):
                    W, b = block[i]
                    dW, db = block_grads[i]
                    block[i] = (W - learning_rate * dW, b - learning_rate * db)

        # Compute epoch's loss
        activations, pre_activations = forward_pass_resnet(X, blocks)
        Y_pred = softmax(pre_activations[-1])
        epoch_loss = cross_entropy_loss(Y_pred, Y)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return blocks, losses

# Initializes a ResNet from a 1-D vector
def initialize_by_vector(vec, N):
    """
    Params:
    vector: 1D array containing weights and biases for the entire network
    N: Number of neurons in any hidden layer of the network

    Returns:
    blocks: List of residual blocks, each containing (W, b) tuples
    """
    blocks = []
    slice_size = N * N + N
    num_layers = len(vec) // slice_size
    num_blocks = num_layers // 3  # Each block has 3 layers
    index = 0
    
    # Initialize residual blocks
    for i in range(num_blocks):

        block = []
        for j in range(3):  # 3 layers

            W = vec[index:index + N * N].reshape(N, N)
            index += N * N

            b = vec[index:index + N].reshape(N, 1)
            index += N

            block.append((W, b))

        blocks.append(block)
    
    # Initialize output layer
    W_out = vec[index:index + N * N].reshape(N, N)
    index += N * N
    b_out = vec[index:index + N].reshape(N, 1)
    
    blocks.append([(W_out, b_out)])  # Output layer is a single block
    
    return blocks

def whole_network_test(X, Y, vec, N):

    # Initialization
    blocks = initialize_by_vector(vec, N)

    # Forward pass
    activations, pre_activations = forward_pass_resnet(X, blocks)

    # Network's output
    Y_pred = softmax(pre_activations[-1])

    # Loss calculation
    loss = cross_entropy_loss(Y_pred, Y)

    return loss, activations, pre_activations, blocks

# Flatten the gradients for all the layers in the network into a single 1-D vector
def flatten_gradients(grads):
    """
    Params:
    grads: List of blocks, each containing tuples (dW, db) representing gradients

    Return:
    flat_vector: A single 1-D array containing all gradients sequentially
    """

    flatten_vector = []
    for block in grads:
        for dW, db, dA in block:
            flatten_vector.append(dW.flatten())  # Flatten weights gradient matrix
            flatten_vector.append(db.flatten())  # Flatten bias gradient vector
    return np.concatenate(flatten_vector)  # Combine all into one vector

def whole_network_gradient(X, Y, activations, pre_activations, blocks):

    grads = backpropagation_resnet(X, Y, activations, pre_activations, blocks)
    return flatten_gradients(grads)


def main():
    """Main function to set up and train the ResNet."""
    # Example data (to be replaced with actual dataset)
    np.random.seed(42)
    input_dim = 20
    hidden_dims = [64, 64, 64]  # Three hidden layers
    output_dim = 10
    num_samples = 1000

    # Example dataset
    X = np.random.randn(input_dim, num_samples)  # Features are rows, samples are columns
    Y = np.zeros((output_dim, num_samples))
    Y[np.random.choice(output_dim, num_samples), np.arange(num_samples)] = 1  # One-hot encoding

    learning_rate = 0.01
    epochs = 50
    batch_size = 64

    blocks, losses = train_resnet_sgd(X, Y, input_dim, hidden_dims, output_dim, learning_rate, epochs, batch_size)

    print("Training completed.")

if __name__ == "__main__":
    main()
