import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape!=y_true.shape:
        return None
    squared_diff = (y_pred-y_true)**2
    MSE=np.mean(squared_diff)
    return MSE
