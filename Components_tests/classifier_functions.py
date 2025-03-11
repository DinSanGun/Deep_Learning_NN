import numpy as np

def softmax(z):
    """
    Compute the softmax of each column of the matrix Z with numerical stability.
    Z: Input matrix of a form: classes x samples
    Returns a softmax probability distribution for each column (sample).
    """
    max_value = np.max(z, axis=0, keepdims=True)
    # Subtract the max value in each column to prevent numerical issues
    exp_z = np.exp(z - max_value)  

    exp_sum = np.sum(exp_z, axis=0, keepdims=True)

    # Normalize by dividing by the sum of the exponentials along each column
    return exp_z / exp_sum

def cross_entropy_loss(Y_pred, Y_true):
    """
    Computes the cross-entropy loss.
    
    Y_pred: Predicted values matrix of size (classes_num , samples_num)
    Y_true: True labels matrix of size (classes_num , samples_num)
    
    Return:
    loss: The softmax cross-entropy loss (scalar)
    """
    samples_num = Y_pred.shape[1] # number of samples being proccessed

    epsilon = 1e-15
    Y_pred_normalized = np.clip(Y_pred, epsilon, 1 - epsilon) # prevent log(epsilon) issues
    
    # Compute the loss 
    loss = - np.sum(Y_true * np.log(Y_pred_normalized)) / samples_num
    return loss

def compute_gradients(X, Y_true, Y_pred):

    # num of samples
    samples_num = Y_pred.shape[1] 

    # Gradient w.r.t. to Y_hat (probs)
    grad_z = (Y_pred - Y_true)

    # Gradient w.r.t. W and b
    grad_W = np.matmul(grad_z, X.T) / samples_num  # Averaging the weights gradient over the number of samples
    grad_b = np.sum(grad_z, axis=1) / samples_num   # Averaging the bias gradient over the number of samples
    
    return grad_W, grad_b