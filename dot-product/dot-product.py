import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x_arr=np.asarray(x, dtype=float)
    y_arr=np.asarray(y, dtype=float)
    
    if x_arr.shape!=y_arr.shape:
        raise ValueError("Array must be same length")
    
    return (np.dot(x_arr, y_arr))