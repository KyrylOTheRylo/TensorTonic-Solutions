import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t: np.ndarray, h_prev: np.ndarray,
             W_r: np.ndarray, W_z: np.ndarray, W_h: np.ndarray,
             b_r: np.ndarray, b_z: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 next hidden state.
    """
    r_t = sigmoid(np.concatenate([h_prev, x_t], axis = -1) @ W_r.T + b_r)
    z_t = sigmoid(np.concatenate([h_prev, x_t], axis = -1) @ W_z.T + b_z)
    h_tilda = np.tanh(np.concatenate([np.multiply(r_t, h_prev), x_t], axis = -1)@W_h.T+ b_h) 
    return np.multiply(z_t, h_prev) + np.multiply((1-z_t), h_tilda)