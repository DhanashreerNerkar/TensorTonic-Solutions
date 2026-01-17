import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    z=np.asarray(x,dtype=float)
    s=1/(1+np.exp(-z))

    
    return s