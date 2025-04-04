import numpy as np

# Creates a multilayer perceptron with layers input 28*28 -> 100 -> ReLU -> 10 -> softmax using only numpy
# Mocked up using ChatGPT

# Activation Functions
def relu(x):
    return np.maximum(0, x)  # ReLU: max(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward Pass Function
def feedforward(X, W1, W2):
    # Add bias to input
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Shape: (batch_size, 785)

    # Hidden Layer
    Z1 = X @ W1  # Matrix multiplication (batch_size, 100)
    H = relu(Z1) # Apply ReLU activation

    # Add bias to hidden layer
    H = np.hstack([H, np.ones((H.shape[0], 1))])  # Shape: (batch_size, 101)

    # Output Layer
    Z2 = H @ W2  # (batch_size, 10)
    Y_pred = softmax(Z2)  # Softmax activation

    return Y_pred

def relu_derivative(x):
    return (x > 0).astype(float)  # Gradient is 1 for x > 0, else 0

# One-hot encoding for labels
def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.size, num_classes))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y

# Cross-entropy loss function
def cross_entropy_loss(Y_pred, Y_true):
    print("np.sum:", np.sum(Y_true * np.log(Y_pred + 1e-9)))
    return -np.mean(np.sum(Y_true * np.log(Y_pred + 1e-9), axis=1))  # Add small epsilon to avoid log(0)

# Alternative sum only defective (for XLNS) Cross-entropy loss function
def cross_entropy_loss_d(Y_pred, Y_true):
    print("np.sum:", np.sum(Y_true * np.log(Y_pred + 1e-9)))
    return -np.sum(Y_true * np.log(Y_pred + 1e-9), axis=1)  # Add small epsilon to avoid log(0)

# Forward and Backward Pass
def train_nn(X, Y, W1, W2, epochs=100, lr=0.01):
    losses = []

    for epoch in range(epochs):
        # ---- FORWARD PASS ----
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias
        Z1 = X_bias @ W1  # Input → Hidden
        H = relu(Z1)  # Apply ReLU activation
        H_bias = np.hstack([H, np.ones((H.shape[0], 1))])  # Add bias to hidden layer
        Z2 = H_bias @ W2  # Hidden → Output
        Y_pred = softmax(Z2)  # Softmax activation

        # Compute Loss
        Y_onehot = one_hot(Y, num_classes=10)  # Convert labels to one-hot
        loss = cross_entropy_loss(Y_pred, Y_onehot)
        losses.append(loss)

        # ---- BACKWARD PASS ----
        # Compute gradient of loss w.r.t. output (Softmax + Cross-Entropy Derivative)
        dL_dZ2 = Y_pred - Y_onehot  # (batch_size, 10)

        # Gradient of W2
        dL_dW2 = H_bias.T @ dL_dZ2  # (101, 10)

        # Backpropagate to hidden layer
        dL_dH = dL_dZ2 @ W2.T  # (batch_size, 101)
        dL_dH = dL_dH[:, :-1]  # Remove bias gradient
        dL_dZ1 = dL_dH * relu_derivative(Z1)  # (batch_size, 100)

        # Gradient of W1
        dL_dW1 = X_bias.T @ dL_dZ1  # (785, 100)

        # ---- UPDATE WEIGHTS ----
        W1 -= lr * dL_dW1  # Update W1
        W2 -= lr * dL_dW2  # Update W2

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    return W1, W2, losses


def predict(X, W1, W2):
    Y_pred_proba = feedforward(X, W1, W2)
    return np.argmax(Y_pred_proba, axis=1)  # Return predicted class