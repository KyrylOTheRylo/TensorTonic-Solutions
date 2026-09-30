import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """
    positions = np.arange(seq_length)[:, None]

    even = np.arange(0, d_model, 2)
    odd = np.arange(1, d_model, 2)

    answer = np.zeros((seq_length, d_model), dtype=float)

    answer[:, even] = np.sin(
        positions / (10000 ** (even / d_model))
    )

    answer[:, odd] = np.cos(
        positions / (10000 ** ((odd - 1) / d_model))
    )

    return answer