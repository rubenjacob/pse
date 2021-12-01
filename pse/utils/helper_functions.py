import torch

EPS = 1e-9


def sample_indices(dim_x: int, sort: bool = False) -> torch.Tensor:
    indices = torch.randperm(dim_x)
    if sort:
        indices = torch.sort(indices)
    return indices


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.unsqueeze(x, dim=1), torch.unsqueeze(y, dim=0)
    similarity_matrix = torch.sum(x * y, dim=-1)
    similarity_matrix /= (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + EPS)
    return similarity_matrix


def torch_gather_nd(params: torch.Tensor, indices: torch.Tensor, out_device: str) -> torch.Tensor:
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn], indices is of 2 dimensions
     and has size [num_samples, m] (m <= n)"""
    params, indices = params.cpu(), indices.cpu()
    return params[indices.transpose(0, 1).long().numpy().tolist()].to(device=out_device)


def torch_scatter_nd_update(tensor: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """tensor has rank 2, indices has size [num_samples, 2], updates has rank n"""
    tensor[indices[:, 0], indices[:, 1]] = updates
    return tensor


def contrastive_loss(similarity_matrix: torch.Tensor, metric_vals: torch.Tensor, temperature: float,
                     coupling_temperature: float, use_coupling_weights: bool, device: str) -> torch.Tensor:
    similarity_matrix /= temperature
    neg_logits1 = similarity_matrix

    col_indices = torch.argmin(metric_vals, dim=1)
    row_indices = torch.arange(0, metric_vals.size()[0]).to(device=col_indices.device)
    pos_indices1 = torch.stack([row_indices, col_indices], dim=1)
    pos_logits1 = torch_gather_nd(similarity_matrix, pos_indices1, out_device=device)

    if use_coupling_weights:
        metric_vals /= coupling_temperature
        coupling = torch.exp(-metric_vals)
        pos_weights1 = -torch_gather_nd(metric_vals, pos_indices1, out_device=device)
        pos_logits1 += pos_weights1
        neg_weights = torch.log((1.0 - coupling) + EPS)
        neg_logits1 += torch_scatter_nd_update(neg_weights, pos_indices1, pos_weights1)

    neg_logits1 = torch.logsumexp(neg_logits1, dim=1)
    return torch.mean(neg_logits1 - pos_logits1)
