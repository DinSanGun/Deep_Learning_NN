import numpy as np

# Provided code (unchanged)
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

    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        W = np.random.randn(hidden_dim, prev_dim) * 0.01 # Initialize with small values
        b = np.zeros((hidden_dim, 1))
        layers.append((W, b))
        prev_dim = hidden_dim

    W_out = np.random.randn(output_dim, prev_dim) * 0.01 # Initialize with small values
    b_out = np.zeros((output_dim, 1))
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
        A = np.tanh(Z)  # Activation
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
            dZ = dA * (1 - np.tanh(Z)**2)  # Derivative of tanh activation

    return grads

def backpropagation_(X, Y, activations, pre_activations_Z, layers):
    samples_num = X.shape[1]  # Number of samples

    # Output layer gradients
    Z4 = pre_activations_Z[-1]
    Y_hat = softmax(Z4)
    dZ4 = (Y_hat - Y) / samples_num  # Jacobian of the loss w.r.t. Z4
    W4, b4 = layers[-1]

    dW4 = dZ4 @ activations[-2].T
    db4 = np.sum(dZ4, axis=1, keepdims=True)

    # Gradients for the third layer
    dA3 = W4.T @ dZ4  # Jacobian of the loss w.r.t. A3 (activation of third layer)
    Z3 = pre_activations_Z[-2]  # Pre-activation of third layer
    J_tanh3 = 1 - np.tanh(Z3)**2  # Derivative of tanh activation for third layer
    dZ3 = dA3 * J_tanh3  # Jacobian of the loss w.r.t. Z3
    W3, b3 = layers[-2]
    dW3 = dZ3 @ activations[-3].T
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    # Gradients for the second layer
    dA2 = W3.T @ dZ3  # Jacobian of the loss w.r.t. A2
    Z2 = pre_activations_Z[-3]  # Pre-activation of 2nd layer
    J_tanh2 = 1 - np.tanh(Z2)**2  # Derivative of tanh activation for 2nd layer
    dZ2 = dA2 * J_tanh2  # Jacobian of the loss w.r.t. Z2
    W2, b2 = layers[-3]
    dW2 = dZ2 @ activations[-4].T
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    # Gradients for the first layer
    dA1 = W2.T @ dZ2  # Jacobian of the loss w.r.t. A1
    Z1 = pre_activations_Z[-4]  # Pre-activation of 1st layer
    J_tanh1 = 1 - np.tanh(Z1)**2  # Derivative of tanh activation for 1st layer
    dZ1 = dA1 * J_tanh1  # Jacobian of the loss w.r.t. Z1
    W1, b1 = layers[-4]
    dW1 = dZ1 @ activations[0].T
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    dX = W1.T @ dZ1 # Derivative of loss w.r.t input matrix

    grads = [(dW1, db1), (dW2, db2), (dW3, db3), (dW4, db4)]
    return grads

# Train the network using SGD
def train_network_sgd(X, Y, input_dim, hidden_dims, output_dim, learning_rate=0.01, epochs=10, batch_size=32):

    # Initialize network's parameters 
    layers = initialize_network(input_dim, hidden_dims, output_dim) # (hidden_dims is an array of sizes)

    num_samples = X.shape[1]  # Number of samples
    losses = []

    for epoch in range(epochs):

        indexes = np.random.permutation(num_samples)
        X = X[:, indexes]  # Shuffle samples
        Y = Y[:, indexes]  # Shuffle labels

        for start in range(0, num_samples, batch_size):

            end = start + batch_size
            X_batch = X[:, start:end]  # Select X batch
            Y_batch = Y[:, start:end]  # Select Y batch

            # Forward pass
            activations, pre_activations_Z = forward_pass(X_batch, layers)
            Y_pred = softmax(pre_activations_Z[-1])  # Output of the network after softmax

            # Compute loss
            loss = cross_entropy_loss(Y_pred, Y_batch)

            # Backpropagation
            grads = backpropagation(X_batch, Y_batch, activations, pre_activations_Z, layers)

            # Update weights and biases
            for i in range(len(layers)):
                W, b = layers[i]
                dW, db = grads[i]
                layers[i] = (W - learning_rate * dW , b - learning_rate * db)

        # Compute epoch's loss
        activations, pre_activations_Z = forward_pass(X, layers)
        Y_pred = softmax(pre_activations_Z[-1])
        epoch_loss = cross_entropy_loss(Y_pred, Y)
        losses.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.3f}")

    return layers, losses

# Initializes weights and biases from a given vector
def initialize_by_vector(vector, N):
    """
    Params:
    vector: 1D array of weights and biases for the entire network
    N: number of neurons in any layer of the network 

    Return:
    layers: List of tuples (W, b), where W is the weight matrix and b is the bias vector of each layer
    """
    layers = []
    slice_size = N * N + N
    num_layers = len(vector) // slice_size
    index = 0

    for _ in range(num_layers):
        W = vector[index:index + N * N].reshape(N, N)
        index += N * N
        b = vector[index:index + N].reshape(N, 1)
        index += N
        layers.append((W, b))

    return layers

# Flatten the gradients for all layers into a single 1D vector
def flatten_gradients(grads):
    """
    Params:
    grads: List of tuples (dW, db) representing gradients for the weights and biases of all the layers

    Return:
    flat_vector: A single 1-D array containing all the gradients of the network sequentially
    """
    flat_vector = []
    for i in range(len(grads)):
        dW, db = grads[i]
        flat_vector.append(dW.flatten())  # Flatten weights gradient matrix
        flat_vector.append(db.flatten())  # Flatten bias gradient vector
    return np.concatenate(flat_vector)  # Combine gradients into one vector

def whole_network_test(X, Y, vec, N):

    # Initialization
    layers = initialize_by_vector(vec, N)

    # Forward pass
    activations, pre_activations = forward_pass(X, layers)

    # Network's output
    Y_pred = softmax(pre_activations[-1])

    # Loss calculation
    loss = cross_entropy_loss(Y_pred, Y)

    return loss, activations, pre_activations, layers

def whole_network_gradient(X, Y, activations, pre_activations, layers):

    grads = backpropagation(X, Y, activations, pre_activations, layers)
    return flatten_gradients(grads)

def forward_pass_i(X, layers, i):
    """
    Params:
    X: Input matrix of shape (features_num x samples_num)
    layers: List of all network's layers, each containing weights and biases
    i: Index of layer to begin forward pass with

    Return:
    activations: List of activations beginning from layer i to the output layer
    pre_activations_Z: List of pre-activations starting from layer i to the output layer
    """

    # Initializing layers before i (so we can use backpropagation as it is - does not affect the test)
    activations = [np.random.randn(*layers[j][0].T.shape) for j in range(i)]
    pre_activations_Z = [np.random.randn(*layers[j][0].shape) for j in range(i)]

    # Start from layer i
    activations.append(X)
    for j in range(i, len(layers) - 1):
        W, b = layers[j]
        Z = W @ activations[-1] + b  # Pre-activation
        pre_activations_Z.append(Z)
        A = np.tanh(Z)  # Activation
        activations.append(A)

    # Output layer
    W_out, b_out = layers[-1]
    Z_out = W_out @ activations[-1] + b_out  # Output layer pre-activation
    pre_activations_Z.append(Z_out)
    activations.append(Z_out)

    return activations, pre_activations_Z

def jacobian_test_layer_i(X, W, b, Y, i, layers):

    # Overwrite initialization in order to give a certain input for testing purposes
    layers[i] = (W,b)

    # Forward pass
    activations, pre_activations = forward_pass_i(X, layers, i)

    Y_hats = softmax(activations[-1])  # Softmax probabilities
    epsilon = 1e-15
    Y_hats_normalized = np.clip(Y_hats, epsilon, 1 - epsilon) # Prevent log issues

    # Compute loss
    loss = cross_entropy_loss(Y_hats_normalized, Y)

    return loss, activations, pre_activations


def main():
    """Main function to set up and train the network."""
    # Example data (to be replaced with actual dataset)
    np.random.seed(37)
    input_dim = 50
    hidden_dims = [32, 32, 16]  # 3 hidden layers
    output_dim = 10
    num_samples = 500

    # Example dataset
    X = np.random.randn(input_dim, num_samples) 
    Y = np.zeros((output_dim, num_samples))
    # Randomly selects a number in range 0 - output_dim for each sample (column)
    Y[np.random.choice(output_dim, num_samples), np.arange(num_samples)] = 1  # One-hot encoding labels matrix

    learning_rate = 0.01
    epochs = 50
    batch_size = 64

    layers, losses = train_network_sgd(X, Y, input_dim, hidden_dims, output_dim, learning_rate, epochs, batch_size)

    print("Training completed.")

if __name__ == "__main__":
    main()
