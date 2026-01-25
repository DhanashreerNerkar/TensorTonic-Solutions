import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    Returns None if the matrix is singular, non-square, or invalid.
    """
    try:
        # 1. Standardize Input (Handle lists and data types)
        # dtype=float ensures precise calculations for determinant/inverse
        matrix = np.asarray(A, dtype=float)

        # 2. Structure Validation
        # Requirement: Must be 2D array
        if matrix.ndim != 2:
            return None
            
        # Requirement: Must be Square (Rows == Cols)
        if matrix.shape[0] != matrix.shape[1]:
            return None
            
        # 3. Singularity Check
        # Calculate Determinant
        det = np.linalg.det(matrix)
        
        # Requirement: Compare against threshold (Hint #2)
        # If determinant is effectively zero, the matrix cannot be inverted.
        if abs(det) < 1e-10:
            return None

        # 4. Compute Inverse
        return np.linalg.inv(matrix)

    except Exception:
        # Catches any edge cases (e.g. empty inputs, malformed lists)
        return None