import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.shape[-1])

    attention_weights = torch.softmax(scores, dim=-1)

    return attention_weights @ V