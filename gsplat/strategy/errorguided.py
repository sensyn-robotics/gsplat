"""Error-guided densification strategy (Spirula-style, EngineDensify.cpp).

The lever isolated as Spirula's edge over gsplat: place capacity where the RENDER IS WRONG, not
where Gaussians are already dense. Each refine step gradually grows the population toward a lean
cap by splitting — along the long axis — the Gaussians with the highest accumulated image-plane
error (the per-Gaussian means2d gradient norm / visibility, gsplat's standard reconstruction-error
signal), plus dead-opacity pruning. No MCMC noise, no unbounded clone/split.

Contrast with the shipped strategies:
- DefaultStrategy: clones/splits by a gradient THRESHOLD (unbounded → over-densifies) + opacity reset.
- MCMCStrategy: relocates/grows by OPACITY-weighted sampling + injected position noise.
Here: gradual growth to a fixed budget, ERROR-weighted long-axis split — matches Spirula's densifier.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from .base import Strategy
from .longaxis_ops import long_axis_split
from .ops import _multinomial_sample, remove


@dataclass
class ErrorGuidedStrategy(Strategy):
    """See module docstring. Reads means2d gradients (set absgrad=True on rasterization)."""

    cap_max: int = 1_000_000
    growth_factor: float = 0.05          # grow +growth_factor * N per refine, toward cap_max
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    absgrad: bool = True
    key_for_gradient: str = "means2d"
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        # grad2d: running sum of per-GS image-plane gradient norm (error). count: visibility count.
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required but missing."

    def step_pre_backward(self, params, optimizers, state, step, info) -> None:
        assert self.key_for_gradient in info, (
            f"{self.key_for_gradient} missing from info; call rasterization(..., absgrad=True)."
        )
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed: bool = False) -> None:
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)
        if step > self.refine_start_iter and step % self.refine_every == 0:
            n_before = len(params["means"])
            n_dead, n_new = self._refine(params, optimizers, state)
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"[errorguided] step {step}: {n_before} -> {len(params['means'])} GS "
                      f"(pruned {n_dead}, split-added {n_new}, cap {self.cap_max})")

    def _update_state(self, params, state, info, packed: bool = False) -> None:
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."
        grads = (info[self.key_for_gradient].absgrad if self.absgrad
                 else info[self.key_for_gradient].grad).clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n, device=grads.device)
        if packed:
            gs_ids = info["gaussian_ids"]                         # [nnz]
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)               # [C, N]
            gs_ids = torch.where(sel)[1]                          # [nnz]
            grads = grads[sel]                                    # [nnz, 2]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

    @torch.no_grad()
    def _refine(self, params, optimizers, state) -> tuple[int, int]:
        device = params["means"].device
        # 1. prune dead (low-opacity) Gaussians — remove() keeps params/optimizers/state consistent.
        dead = torch.sigmoid(params["opacities"]) < self.min_opacity
        n_dead = int(dead.sum())
        if n_dead > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=dead)
        # 2. gradual growth to cap_max by ERROR-weighted long-axis split.
        n = len(params["means"])
        if n >= self.cap_max:
            return n_dead, 0
        n_add = min(self.cap_max, math.ceil((1.0 + self.growth_factor) * n)) - n
        if n_add <= 0:
            return n_dead, 0
        error = state["grad2d"] / state["count"].clamp_min(1)     # mean image-plane error per GS
        if float(error.sum()) <= 0:                               # no visibility yet — skip
            return n_dead, 0
        idx = _multinomial_sample(error, min(n_add, n), replacement=False)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[idx] = True
        long_axis_split(params=params, optimizers=optimizers, state=state, mask=mask)
        return n_dead, int(mask.sum())
