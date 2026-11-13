"""Harmonic (structural) smoothing of a bi-temporal feature pair on one shared graph.

Feature-level counterpart of the advisor's MATLAB demo: the nodes are the cells of the
CNN feature map, the 8-neighbour affinity graph is built from the mean of both dates,
and each date is replaced by its structural component
    S = argmin 0.5*||S - F||^2 + alpha/2 * trace(S' L S)  =>  (I + alpha*L) S = F.
"""
import torch
from torch.nn import functional as F


def shared_laplacian(z):
    """Symmetric normalized Laplacian (batch, h*w, h*w) of the 8-neighbour graph on z (batch, c, h, w)."""
    batch, _, h, w = z.shape
    n = h * w
    z = (z - z.mean(dim=(2, 3), keepdim=True)) / (z.std(dim=(2, 3), keepdim=True) + 1e-6)

    # Squared feature distance of every node to its right, lower, lower-right and lower-left
    # neighbour. `pad` (left, right, top, bottom) puts each edge at its source node on the
    # full grid; `offset` is the neighbour's distance in row-major node order.
    edges = [
        (z[:, :, :, :-1] - z[:, :, :, 1:], (0, 1, 0, 0), 1),
        (z[:, :, :-1, :] - z[:, :, 1:, :], (0, 0, 0, 1), w),
        (z[:, :, :-1, :-1] - z[:, :, 1:, 1:], (0, 1, 0, 1), w + 1),
        (z[:, :, :-1, 1:] - z[:, :, 1:, :-1], (1, 0, 0, 1), w - 1),
    ]
    distances = [difference.pow(2).sum(dim=1) for difference, _, _ in edges]
    # Median by sorting: torch.median has no deterministic CUDA implementation.
    ordered = torch.cat([d.flatten(1) for d in distances], dim=1).sort(dim=1).values
    sigma2 = ordered[:, (ordered.size(1) - 1) // 2].clamp_min(1e-6)

    # Missing edges are padded with zero weight, so each direction fills one diagonal.
    upper = z.new_zeros(batch, n, n)
    for distance, (_, pad, offset) in zip(distances, edges):
        weight = F.pad(torch.exp(-distance / (2 * sigma2.view(-1, 1, 1))), pad).flatten(1)
        upper = upper + torch.diag_embed(weight[:, :n - offset], offset=offset)
    affinity = upper + upper.transpose(1, 2)

    scale = affinity.sum(dim=2).clamp_min(1e-8).rsqrt()
    return torch.eye(n, device=z.device, dtype=z.dtype) - scale.unsqueeze(2) * affinity * scale.unsqueeze(1)


def harmonic_smooth(feat1, feat2, alpha):
    """Structural components of two feature maps (batch, c, h, w); alpha=0 returns them unchanged."""
    if alpha == 0:
        return feat1, feat2
    n = feat1.size(2) * feat1.size(3)
    system = torch.eye(n, device=feat1.device, dtype=feat1.dtype) + alpha * shared_laplacian(0.5 * (feat1 + feat2))
    stacked = torch.cat([feat1, feat2], dim=1)
    smooth = torch.linalg.solve(system, stacked.flatten(2).transpose(1, 2))  # (batch, h*w, 2c)
    return smooth.transpose(1, 2).reshape(stacked.shape).chunk(2, dim=1)
