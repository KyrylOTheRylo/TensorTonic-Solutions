import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    positions = np.arange(seq_len)[:,None]          # (seq_len, 1)
    dims_even = np.arange(0, d_model, 2) 
    dims_odd = np.arange(1, d_model, 2) 

    answer  = np.zeros((seq_len, d_model), dtype = float)
    answer[:, dims_even] = np.sin(positions/ (base ** (dims_even/d_model)))
    answer[:, dims_odd] = np.cos(positions/ (base ** ((dims_odd-1)/d_model)))

    return answer