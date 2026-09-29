import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x_new = np.array(x)
    sample = np.random.random(x_new.shape)
    if rng:
        sample = rng.random(x_new.shape)
    mask = np.where(sample>= (1-p), 0 , 1/(1-p) )
    answer = x * mask    
    return (answer, mask)