import numpy as np
import matplotlib.pyplot as plt

def sgd_least_squares(A, b, lr = 0.01, epochs = 10, batch_size = 32):
    """
    Perform Stochastic Gradient Descent (SGD) to minimize a least squares cost function.
    
    Args:
    - A: Input matrix
    - b: target vector
    - lr: Learning rate for SGD.
    - epochs: Number of epochs to run.
    - batch_size: Number of samples per batch.
    
    Returns:
    - x: Solution vector
    - losses: List of loss values after each epoch.
    """
    # num of observations
    n_samples = A.shape[0] 

    # num of predictors per observation
    n_features = A.shape[1]

    # Randomize parameters vector
    x = np.random.randn(n_features)  
    
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data samples in the beginning of epoch
        indexes = np.random.permutation(n_samples)

        # Reorder input and true label matrices
        A = A[indexes]
        b = b[indexes]

        # Mini-batch processing
        for start in range(0, n_samples, batch_size):

            end = start + batch_size
            
            # Select mini-batch
            A_batch = A[start:end]
            b_batch = b[start:end]
            
            # Prediction
            b_pred = A_batch @ x 

            # Loss calculation
            residual = b_pred - b_batch 

            # Gradient of loss w.r.t weights
            grad_x = A_batch.T @ residual
            
            # Update parameters
            x = x - lr * grad_x
                    
        residual = (A_batch @ x) - b_batch
        epoch_loss = (1 / (2 * n_samples)) * np.sum(residual**2)  # Sum of squares residuals

        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return x, losses

# Plotting results
def plot_results_with_data_points(losses):
    """
    Plot the regression line, true data points, and training loss with data points on the graph.
    """
    plt.figure(figsize=(5, 4))
    
    # Plot 2: Loss Curve with Data Points
    plt.plot(range(1, len(losses) + 1), losses, label="Loss", color="green", linewidth=2)
    plt.scatter(range(1, len(losses) + 1), losses, color="red", label="Loss Data Points", s=15)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve with Data Points")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def setup(N, M): 

    # N - number of observations
    # M - number of features
    A = np.random.randn(N, M)

    # Generate true values parameter vector
    x_true = np.random.randn(M)  

    # Add noise to the true values vector
    b = (A @ x_true) + (0.25 * np.random.randn(N))

    return A, b

# Example usage
if __name__ == "__main__":

    # Generate synthetic least squares data
    # 15 observations (samples), 4 predictors (features)
    A, b = setup(15, 4) 
    
    # Perform SGD
    learned_x, losses = sgd_least_squares(A, b, lr = 0.01, epochs = 15, batch_size = 5)

    # Print results
    print("\nLearned x:", learned_x)
    
    # Plot results
    plot_results_with_data_points(losses)