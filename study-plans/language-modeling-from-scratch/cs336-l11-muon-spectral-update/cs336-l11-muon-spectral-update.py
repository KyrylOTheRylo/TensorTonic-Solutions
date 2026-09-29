import torch

def muon_spectral_update(
    parameter: torch.Tensor, gradient: torch.Tensor,
    previous_momentum: torch.Tensor, momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors: new_parameter, new_momentum, orthogonalized_update.
    """
    
    B = momentum_coefficient * previous_momentum + gradient

    # 2. SVD must be done in float32
    B_float = B.float()

    # 3. Compact SVD
    U, S, Vh = torch.linalg.svd(
        B_float,
        full_matrices=False
    )

    # 4. Replace singular values with 1
    O_t = U @ Vh

    # 5. Convert spectral update back to parameter dtype
    O_t = O_t.to(parameter.dtype)

    # 6. Parameter update
    W_new = parameter - learning_rate * O_t

    return {
        "new_parameter": W_new,
        "new_momentum": B,
        "orthogonalized_update": O_t
    }