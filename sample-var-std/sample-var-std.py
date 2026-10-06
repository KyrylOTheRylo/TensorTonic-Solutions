import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    arr = np.asarray(x, dtype = float)
    std = np.std(arr, ddof = 1)
    return {'variance': float(std**2), 'standard_deviation': float(std)}