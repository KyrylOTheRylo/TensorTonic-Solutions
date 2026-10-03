import numpy as np

def hidden_state_update(z_t: np.ndarray, h_prev: np.ndarray,
                        h_tilde: np.ndarray) -> np.ndarray:
    """
    Returns the float64 updated hidden state.
    """
    return np.multiply(z_t, h_prev) + np.multiply((1 - z_t), h_tilde) 