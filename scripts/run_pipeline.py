#!/usr/bin/env python3
"""Orchestrate COLMAP refinement + gsplat training + evaluation in one shot.

Usage:
    uv run python scripts/run_pipeline.py --config scripts/pipeline_config.yaml
    uv run python scripts/run_pipeline.py --config <path> --force
    uv run python scripts/run_pipeline.py --config <path> --dry-run

Stages (each skipped if its output already exists, unless --force):
    1. Average input intrinsics into one shared PINHOLE camera.
    2. Rewrite cameras.txt / images.txt under output_dir/sparse/0/.
    3. pycolmap: extract → match (sequential) → triangulate → bundle adjust.
    4. (optional) Append LIDAR points to refined points3D.txt.
    5. gsplat training via Config(...) + cli(main, cfg).
    6. Read final val_step{max_steps-1}.json; write pipeline_results.json.

All outputs live under paths.output_dir. The input dataset is never modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Make sibling modules importable when running from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Register the pycolmap.SceneManager adapter BEFORE anything imports gsplat.
import scene_manager_compat  # noqa: E402
scene_manager_compat.install()

from intrinsics_utils import (  # noqa: E402
    AveragedIntrinsics,
    average_intrinsics,
    concat_lidar_points,
    rewrite_images_to_single_camera,
    write_single_camera_txt,
)
from colmap_refine import ColmapConfig, refine  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_workspace(input_colmap: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sparse/0").mkdir(parents=True, exist_ok=True)

    images_link = output_dir / "images"
    src_images = input_colmap / "images"
    if images_link.is_symlink() or images_link.exists():
        if images_link.is_symlink():
            target = Path(os.readlink(images_link))
            if target.resolve() == src_images.resolve():
                pass  # already correct
            else:
                images_link.unlink()
                images_link.symlink_to(src_images)
        # If it's a real dir, leave it alone (user might have populated).
    else:
        images_link.symlink_to(src_images)

    return output_dir / "sparse/0", output_dir / "colmap.db"


def stage_intrinsics(input_sparse: Path, output_sparse: Path, cfg: dict, force: bool) -> AveragedIntrinsics:
    cameras_out = output_sparse / "cameras.txt"
    images_out = output_sparse / "images.txt"

    intr = average_intrinsics(input_sparse / "cameras.txt", max_relative_stdev=cfg["max_relative_stdev"])
    print(
        f"[intrinsics] averaged {intr.n_input_cameras} cameras: "
        f"fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.cx:.3f} cy={intr.cy:.3f} "
        f"rel_stdev={ {k: f'{v:.4%}' for k, v in intr.rel_stdev.items()} }"
    )

    if force or not cameras_out.exists():
        write_single_camera_txt(cameras_out, intr)
        print(f"[intrinsics] wrote {cameras_out}")
    else:
        print(f"[intrinsics] {cameras_out} exists, skipping (use --force to overwrite)")

    if force or not images_out.exists():
        n = rewrite_images_to_single_camera(input_sparse / "images.txt", images_out)
        print(f"[intrinsics] rewrote {n} image rows to {images_out} with camera_id=1")
    else:
        print(f"[intrinsics] {images_out} exists, skipping")

    return intr


def stage_colmap(
    output_sparse: Path,
    image_dir: Path,
    database_path: Path,
    intrinsics: AveragedIntrinsics,
    cfg: dict,
    force: bool,
):
    refined_points = output_sparse / "points3D.txt"
    # We consider COLMAP done if there's a non-empty points3D.txt.
    if not force and refined_points.exists() and refined_points.stat().st_size > 200:
        with open(refined_points) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    print(f"[colmap] {refined_points} non-empty, skipping (use --force)")
                    return None

    priors_csv = cfg.get("position_priors_csv")
    if priors_csv is not None:
        priors_csv = Path(priors_csv).expanduser().resolve()
    colmap_cfg = ColmapConfig(
        max_num_features=cfg["max_num_features"],
        use_gpu=cfg["use_gpu"],
        seq_overlap=cfg["seq_overlap"],
        seq_quadratic_overlap=cfg["seq_quadratic_overlap"],
        loop_detection=cfg["loop_detection"],
        refine_focal_length=cfg["refine_focal_length"],
        refine_principal_point=cfg["refine_principal_point"],
        refine_extra_params=cfg["refine_extra_params"],
        refine_extrinsics=cfg["refine_extrinsics"],
        refine_points3D=cfg["refine_points3D"],
        ba_max_iterations=cfg["ba_max_iterations"],
        position_priors_csv=priors_csv,
        position_prior_variance=float(cfg.get("position_prior_variance", 0.01)),
    )
    return refine(
        image_dir=image_dir,
        sparse_in=output_sparse,
        sparse_out=output_sparse,
        database_path=database_path,
        intrinsics=intrinsics,
        cfg=colmap_cfg,
        force=force,
    )


def stage_lidar(input_colmap: Path, output_sparse: Path, cfg: dict, force: bool) -> tuple[int, int] | None:
    if not cfg["concat_into_refined"]:
        print("[lidar] disabled, skipping")
        return None
    refined_points = output_sparse / "points3D.txt"
    lidar_source = input_colmap / cfg["source"]
    if not lidar_source.exists():
        print(f"[lidar] source {lidar_source} not found, skipping")
        return None

    sentinel = output_sparse / ".lidar_concat_done"
    if not force and sentinel.exists():
        print("[lidar] already concatenated (sentinel exists), skipping")
        return None

    n_refined, n_lidar = concat_lidar_points(lidar_source, refined_points, refined_points)
    sentinel.touch()
    print(f"[lidar] appended {n_lidar} LIDAR points to {n_refined} feature points")
    return n_refined, n_lidar


def _gpu_free_mb(device_index: int = 0) -> int | None:
    """Query free memory on the GPU via nvidia-smi. None if smi is unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.splitlines()[0])
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


def _wait_for_gpu(min_free_mb: int, poll_seconds: int, timeout_seconds: int, device_index: int = 0) -> None:
    """Block until `min_free_mb` is free on the GPU, or raise on timeout.

    The previous OOM happened because another training was holding 10.5 GB of
    11.6 GB at the exact moment we tried to start gsplat — leaving only 20 MB
    free. This pre-flight check ensures we never launch into a clearly
    insufficient memory state.
    """
    if min_free_mb <= 0:
        return  # disabled
    deadline = time.monotonic() + max(1, timeout_seconds)
    first = True
    while True:
        free = _gpu_free_mb(device_index)
        if free is None:
            print(f"[gsplat] nvidia-smi unavailable; skipping GPU memory pre-flight")
            return
        if free >= min_free_mb:
            print(f"[gsplat] GPU{device_index} free={free} MiB >= required {min_free_mb} MiB; proceeding")
            return
        if first:
            print(
                f"[gsplat] GPU{device_index} free={free} MiB < required {min_free_mb} MiB; "
                f"waiting (poll every {poll_seconds}s, timeout {timeout_seconds}s)"
            )
            first = False
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"GPU{device_index} still has only {free} MiB free after waiting "
                f"{timeout_seconds}s (needed {min_free_mb}); aborting before OOM."
            )
        time.sleep(max(1, poll_seconds))


def stage_gsplat(output_dir: Path, cfg: dict, runtime_cfg: dict, force: bool) -> dict:
    gsplat_result_dir = output_dir / "gsplat"
    final_stats = gsplat_result_dir / "stats" / f"val_step{cfg['max_steps'] - 1:04d}.json"
    if not force and final_stats.exists():
        print(f"[gsplat] final stats exist at {final_stats}, skipping training")
        return _load_metrics(final_stats)

    os.environ["CUDA_VISIBLE_DEVICES"] = runtime_cfg["cuda_visible_devices"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = runtime_cfg["alloc_conf"]

    # Pre-flight: don't start torch + gsplat if the GPU is starved by another
    # process. This guards against the OOM we hit on a shared GPU. Disable by
    # setting `runtime.min_free_gpu_mb: 0` in the YAML.
    device_index = int(str(runtime_cfg["cuda_visible_devices"]).split(",")[0])
    min_free_mb = int(runtime_cfg.get("min_free_gpu_mb", 5000))
    _wait_for_gpu(
        min_free_mb=min_free_mb,
        poll_seconds=int(runtime_cfg.get("gpu_poll_seconds", 30)),
        timeout_seconds=int(runtime_cfg.get("gpu_wait_timeout_seconds", 7200)),
        device_index=device_index,
    )

    # Race-condition guard: a competing process could grab memory between
    # the check and our first real allocation. Claim our share NOW by
    # allocating a tensor and letting it fall out of scope — PyTorch's
    # caching allocator keeps the bytes in-process so other processes can't
    # reach them. (Crucially we do NOT call `torch.cuda.empty_cache()`; that
    # would release back to the driver and undo the reservation.)
    # Disable by setting `runtime.reserve_gpu_mb: 0`.
    reserve_mb = int(runtime_cfg.get("reserve_gpu_mb", min_free_mb))
    if reserve_mb > 0:
        try:
            import torch  # late import: avoid CUDA init until after the wait
            n_floats = reserve_mb * 1024 * 1024 // 4
            _hold = torch.empty(n_floats, dtype=torch.float32, device=f"cuda:{device_index}")
            del _hold  # returns to torch's cache, NOT to the CUDA driver
            print(f"[gsplat] claimed {reserve_mb} MiB in PyTorch's CUDA cache")
        except Exception as e:
            print(f"[gsplat] memory reservation skipped ({type(e).__name__}: {e})")

    # Late imports so env vars apply before torch sees the GPU.
    examples_dir = SCRIPT_DIR.parent / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from simple_trainer import Config, main  # noqa: E402
    from gsplat.distributed import cli  # noqa: E402
    from gsplat.strategy import DefaultStrategy, MCMCStrategy  # noqa: E402

    strategy_name = cfg["strategy"].lower()
    if strategy_name == "default":
        strategy = DefaultStrategy(verbose=True)
    elif strategy_name == "mcmc":
        strategy = MCMCStrategy(verbose=True, cap_max=int(cfg["mcmc_cap_max"]))
    else:
        raise ValueError(f"Unknown strategy: {cfg['strategy']}")

    gs_cfg = Config(
        data_dir=str(output_dir),
        data_factor=int(cfg["data_factor"]),
        result_dir=str(gsplat_result_dir),
        max_steps=int(cfg["max_steps"]),
        eval_steps=list(cfg["eval_steps"]),
        save_steps=list(cfg["save_steps"]),
        test_every=int(cfg["test_every"]),
        sh_degree=int(cfg["sh_degree"]),
        normalize_world_space=bool(cfg["normalize_world_space"]),
        pose_opt=bool(cfg["pose_opt"]),
        pose_opt_lr=float(cfg["pose_opt_lr"]),
        disable_viewer=bool(cfg["disable_viewer"]),
        port=int(cfg["port"]),
        strategy=strategy,
        # Trajectory video render at the end can fail when the SfM is partial
        # (only some images registered); the trained model + metrics are
        # already saved before this step, so we skip the optional video.
        disable_video=True,
        # Always emit PLY at each save_step so the gaussians are immediately
        # openable in SuperSplat / MeshLab / CloudCompare without manual
        # ckpt-to-ply conversion.
        save_ply=True,
    )
    gs_cfg.adjust_steps(gs_cfg.steps_scaler)

    print(
        f"[gsplat] launching training: max_steps={gs_cfg.max_steps}, "
        f"strategy={strategy_name}, pose_opt={gs_cfg.pose_opt}, "
        f"disable_viewer={gs_cfg.disable_viewer}"
    )
    cli(main, gs_cfg, verbose=True)
    return _load_metrics(final_stats)


def _load_metrics(stats_json: Path) -> dict:
    if not stats_json.exists():
        return {"error": f"missing {stats_json}"}
    with open(stats_json) as f:
        return json.load(f)


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="re-run all stages, ignore existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="parse config + validate paths only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override output dir; bypasses the dataname_<timestamp> auto-suffix. "
        "Use this to resume an existing run (will then honor skip-if-exists).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_colmap = Path(cfg["paths"]["input_colmap"]).resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        base = Path(cfg["paths"]["output_dir"]).resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base.with_name(f"{base.name}_{stamp}")

    if not (input_colmap / "images").exists():
        print(f"ERROR: {input_colmap}/images not found", file=sys.stderr)
        return 1
    if not (input_colmap / "sparse/0/cameras.txt").exists():
        print(f"ERROR: {input_colmap}/sparse/0/cameras.txt not found", file=sys.stderr)
        return 1

    print(f"input_colmap: {input_colmap}")
    print(f"output_dir:   {output_dir}")
    if args.dry_run:
        print("dry-run OK")
        return 0

    output_sparse, database_path = prepare_workspace(input_colmap, output_dir)

    intr = stage_intrinsics(input_colmap / "sparse/0", output_sparse, cfg["intrinsics"], args.force)
    refinement = stage_colmap(output_sparse, output_dir / "images", database_path, intr, cfg["colmap"], args.force)
    lidar_counts = stage_lidar(input_colmap, output_sparse, cfg["lidar"], args.force)
    metrics = stage_gsplat(output_dir, cfg["gsplat"], cfg["runtime"], args.force)

    summary = {
        "input_colmap": str(input_colmap),
        "output_dir": str(output_dir),
        "intrinsics": dataclasses.asdict(intr),
        "colmap_refinement": dataclasses.asdict(refinement) if refinement is not None else "skipped",
        "lidar_concat": {"n_refined": lidar_counts[0], "n_lidar": lidar_counts[1]} if lidar_counts else "skipped",
        "final_metrics": metrics,
        "gsplat_config": cfg["gsplat"],
    }
    out_json = output_dir / cfg["eval"]["out_json"]
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[done] summary -> {out_json}")
    print("\nFinal metrics:")
    for k in ("psnr", "ssim", "lpips", "num_GS", "ellipse_time"):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
