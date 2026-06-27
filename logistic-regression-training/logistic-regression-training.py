import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Convert inputs to NumPy arrays in case they are passed as standard lists
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Extract dimensions (N = number of samples, D = number of features)
    N, D = X.shape
    
    # Initialize parameters as per requirements
    w = np.zeros(D)
    b = 0.0
    
    # Gradient Descent loop
    for _ in range(steps):
        # 1. Forward pass (compute predictions)
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 2. Compute gradients
        # dw shape will be (D,), same as w
        dw = np.dot(X.T, (p - y)) / N 
        db = np.mean(p - y)
        
        # 3. Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b