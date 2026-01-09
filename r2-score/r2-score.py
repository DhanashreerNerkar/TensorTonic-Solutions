import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)

    if ss_tot == 0.0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    return float(1.0 - ss_res / ss_tot)
