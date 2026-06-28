import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    
    # Requirement: If seqs is empty, return an array of shape (0, 0)
    if N == 0:
        return np.empty((0, 0), dtype=int)
        
    # Determine the target sequence length
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len
        
    # Initialize the result array filled entirely with the pad_value
    # We specify dtype=int as per the requirements
    result = np.full((N, L), fill_value=pad_value, dtype=int)
    
    # Copy each sequence into the result array
    for i, seq in enumerate(seqs):
        # Figure out how much of the sequence we can actually fit
        length_to_copy = min(len(seq), L)
        
        # If there is data to copy, place it in the row and truncate if necessary
        if length_to_copy > 0:
            result[i, :length_to_copy] = seq[:length_to_copy]
            
    return result