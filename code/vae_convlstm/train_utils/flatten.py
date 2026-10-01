import numpy as np
import torch

def numpy_flatten(x: np.ndarray, start_dim=0, end_dim=-1)-> np.ndarray:
    shape = x.shape
    if end_dim < 0:
        end_dim += len(shape)
    flattened_dim = int(np.prod(shape[start_dim:end_dim + 1]))
    new_shape = shape[:start_dim] + (flattened_dim,) + shape[end_dim + 1:]
    return x.reshape(new_shape)

def torch_flatten(x: torch.Tensor, start_dim=0, end_dim=-1)-> torch.Tensor:
    return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)