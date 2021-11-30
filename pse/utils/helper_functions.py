import time

import torch

EPS = 1e-9


def sample_indices(dim_x: int, size: int = 128, sort: bool = False) -> torch.Tensor:
    indices = torch.randperm(dim_x)[:size]
    if sort:
        indices = torch.sort(indices)
    return indices


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.unsqueeze(x, dim=1), torch.unsqueeze(y, dim=0)
    similarity_matrix = torch.sum(x * y, dim=-1)
    similarity_matrix /= (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + EPS)
    return similarity_matrix


def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn], indices is of 2 dimensions
     and has size [num_samples, m] (m <= n)"""
    return params[indices.transpose(0, 1).long()]


def torch_scatter_nd_update(tensor: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """tensor has rank 2, indices has size [num_samples, 2], updates has rank n"""
    tensor[indices[:, 0], indices[:, 1]] = updates
    return tensor
