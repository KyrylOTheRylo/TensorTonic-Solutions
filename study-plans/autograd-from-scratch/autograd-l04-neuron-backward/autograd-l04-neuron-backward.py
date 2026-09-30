import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = weights @ inputs + bias
    y = torch.tanh(a)
    delta = upstream_gradient * (1-y**2)
    dl_dx = delta*weights
    dl_dw = delta*inputs
    dl_db = delta
    return (y, dl_dx, dl_dw, dl_db)