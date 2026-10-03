import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(
            hidden_dim, hidden_dim + input_dim
        ) * scale

        self.W_z = np.random.randn(
            hidden_dim, hidden_dim + input_dim
        ) * scale

        self.W_h = np.random.randn(
            hidden_dim, hidden_dim + input_dim
        ) * scale

        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(
            output_dim, hidden_dim
        ) * np.sqrt(2.0 / (hidden_dim + output_dim))

        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> dict:
        """
        X shape:
            (batch_size, sequence_length, input_dim)

        Returns:
            {
                "outputs": (batch_size, sequence_length, output_dim),
                "final_hidden_state": (batch_size, hidden_dim)
            }
        """

        batch_size = X.shape[0]
        sequence_length = X.shape[1]

        h_prev = np.zeros(
            (batch_size, self.hidden_dim),
            dtype=np.float64
        )

        outputs = []

        for t in range(sequence_length):

            x_t = X[:, t, :]

            combined = np.concatenate(
                [h_prev, x_t],
                axis=-1
            )

            # Reset gate
            r_t = sigmoid(
                combined @ self.W_r.T + self.b_r
            )

            # Update gate
            z_t = sigmoid(
                combined @ self.W_z.T + self.b_z
            )

            # Candidate hidden state
            candidate_input = np.concatenate(
                [r_t * h_prev, x_t],
                axis=-1
            )

            h_tilde = np.tanh(
                candidate_input @ self.W_h.T + self.b_h
            )

            # New hidden state
            h_t = (
                z_t * h_prev
                + (1.0 - z_t) * h_tilde
            )

            # Output projection
            y_t = h_t @ self.W_y.T + self.b_y

            outputs.append(y_t)

            h_prev = h_t

        outputs = np.stack(outputs, axis=1)

        return {
            "outputs": outputs.astype(np.float64),
            "final_hidden_state": h_prev.astype(np.float64)
        }