import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    a = weight_matrix @ inputs.T + biases
    print(a)
    if nonlinear:
        return torch.tanh(a)
    else:
        return a
