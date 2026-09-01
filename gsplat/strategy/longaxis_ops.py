"""Long-axis Gaussian split (Spirula-style densify.slang:85-104).

Splits each selected Gaussian into two COMPACT children along its longest principal axis — the long
axis shrunk x0.5, the other two x0.85, children offset +/-0.5*(longest scale) along that axis — vs
gsplat's isotropic split (ops.split) which samples from the full covariance and leaves floaters.
Mirrors ops.split's param/optimizer/state surgery; only the geometry differs. Net +1 per split."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils import normalized_quat_to_rotmat
from .ops import _update_param_with_optimizer


@torch.no_grad()
def long_axis_split(params, optimizers, state, mask: Tensor) -> None:
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    n = len(sel)
    ar = torch.arange(n, device=device)

    scales_lin = torch.exp(params["scales"][sel])                 # [n,3]
    rot = normalized_quat_to_rotmat(F.normalize(params["quats"][sel], dim=-1))  # [n,3,3]
    long_idx = scales_lin.argmax(dim=1)                           # [n]
    axis = rot[ar, :, long_idx]                                   # [n,3] world dir of the long axis
    offset = axis * (0.5 * scales_lin[ar, long_idx])[:, None]     # [n,3]
    new_log = params["scales"][sel] + torch.log(torch.tensor(0.85, device=device))
    new_log[ar, long_idx] = params["scales"][sel][ar, long_idx] + torch.log(
        torch.tensor(0.5, device=device))

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            m = p[sel]
            children = torch.stack([m + offset, m - offset], dim=0).reshape(-1, 3)  # [2n,3]
        elif name == "scales":
            children = new_log.repeat(2, 1)                       # [2n,3] both children same shape
        else:
            children = p[sel].repeat([2] + [1] * (p.dim() - 1))
        return torch.nn.Parameter(torch.cat([p[rest], children]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v[rest], torch.zeros((2 * n, *v.shape[1:]), device=device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat([v[rest], v[sel].repeat([2] + [1] * (v.dim() - 1))])
