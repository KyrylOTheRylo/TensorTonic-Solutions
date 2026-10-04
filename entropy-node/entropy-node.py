import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    y= np.asarray(y, dtype= np.int64)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)