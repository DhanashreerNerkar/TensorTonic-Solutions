import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
   # 1. Convert to a numpy array to prevent List errors
    x_arr = np.array(x, dtype=float)
    
    # 2. Clip the values to avoid exponential overflow (bounds between -500 and 500)
    x_arr = np.clip(x_arr, -500, 500)
    
    # 3. Calculate and return
    return 1 / (1 + np.exp(-x_arr))