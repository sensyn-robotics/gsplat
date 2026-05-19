"""Average per-frame iPhone intrinsics into one shared PINHOLE camera and
rewrite cameras.txt / images.txt so every image references camera_id=1.

The iPhone's physical lens is fixed, so the per-frame variation in
intrinsics.csv (stdev < 0.25% relative) is measurement noise. Collapsing to
one shared camera unlocks COLMAP's `single_camera=1` path and makes bundle
adjustment well-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AveragedIntrinsics:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    n_input_cameras: int
    rel_stdev: dict[str, float]


def average_intrinsics(cameras_txt: Path, max_relative_stdev: float = 0.01) -> AveragedIntrinsics:
    fxs, fys, cxs, cys = [], [], [], []
    models, widths, heights = set(), set(), set()
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            models.add(parts[1])
            widths.add(int(parts[2]))
            heights.add(int(parts[3]))
            fxs.append(float(parts[4]))
            fys.append(float(parts[5]))
            cxs.append(float(parts[6]))
            cys.append(float(parts[7]))

    if len(models) != 1:
        raise ValueError(f"Mixed camera models in {cameras_txt}: {models}")
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError(f"Mixed image dimensions: widths={widths}, heights={heights}")
    model = next(iter(models))
    if model != "PINHOLE":
        raise ValueError(f"Only PINHOLE is supported by intrinsic averaging (got {model})")

    fx_a = np.array(fxs); fy_a = np.array(fys); cx_a = np.array(cxs); cy_a = np.array(cys)

    def rel_std(a):
        return float(a.std() / a.mean())

    rel = {"fx": rel_std(fx_a), "fy": rel_std(fy_a), "cx": rel_std(cx_a), "cy": rel_std(cy_a)}
    worst = max(rel.values())
    if worst > max_relative_stdev:
        raise ValueError(
            f"Relative stdev {worst:.4%} exceeds threshold {max_relative_stdev:.4%}; "
            f"per-axis: {rel}. Averaging would discard real intrinsic variation."
        )

    return AveragedIntrinsics(
        model=model,
        width=next(iter(widths)),
        height=next(iter(heights)),
        fx=float(fx_a.mean()),
        fy=float(fy_a.mean()),
        cx=float(cx_a.mean()),
        cy=float(cy_a.mean()),
        n_input_cameras=len(fxs),
        rel_stdev=rel,
    )


def write_single_camera_txt(out_path: Path, intr: AveragedIntrinsics, camera_id: int = 1) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        f"{camera_id} {intr.model} {intr.width} {intr.height} "
        f"{intr.fx:.6f} {intr.fy:.6f} {intr.cx:.6f} {intr.cy:.6f}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def rewrite_images_to_single_camera(images_in: Path, images_out: Path, camera_id: int = 1) -> int:
    """Copy every image row, replacing its per-frame camera_id with `camera_id`.
    The POINTS2D line (line 2 per image) is preserved verbatim (typically empty).
    Returns number of images written.
    """
    images_out.parent.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    n = 0
    with open(images_in) as f:
        expect_cam_line = True
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith("#"):
                out_lines.append(line)
                continue
            if not stripped:
                out_lines.append(line)
                if not expect_cam_line:
                    expect_cam_line = True
                continue
            if expect_cam_line:
                parts = stripped.split()
                parts[8] = str(camera_id)
                out_lines.append(" ".join(parts))
                expect_cam_line = False
                n += 1
            else:
                out_lines.append(line)
                expect_cam_line = True

    if n > 0 and not out_lines:
        out_lines = [
            "# Image list with two lines of data per image:",
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "#   POINTS2D[] as (X, Y, POINT3D_ID)",
            f"# Number of images: {n}",
        ]
    images_out.write_text("\n".join(out_lines) + "\n")
    return n


def concat_lidar_points(
    lidar_txt: Path,
    refined_txt: Path,
    out_txt: Path,
) -> tuple[int, int]:
    """Append LIDAR-derived points from `lidar_txt` to feature-triangulated
    points in `refined_txt`, offsetting LIDAR IDs so they don't collide.
    Returns (n_refined, n_lidar_appended)."""

    refined_lines: list[str] = []
    n_refined = 0
    max_id = 0
    if refined_txt.exists():
        with open(refined_txt) as f:
            for raw in f:
                s = raw.rstrip("\n")
                stripped = s.strip()
                if stripped.startswith("#") or not stripped:
                    refined_lines.append(s)
                    continue
                pid = int(stripped.split()[0])
                max_id = max(max_id, pid)
                refined_lines.append(s)
                n_refined += 1

    lidar_rows: list[str] = []
    if lidar_txt.exists():
        with open(lidar_txt) as f:
            for raw in f:
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                new_id = max_id + 1 + len(lidar_rows)
                parts[0] = str(new_id)
                # LIDAR has no observation track; keep empty TRACK[].
                lidar_rows.append(" ".join(parts[:8]))

    header = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {n_refined + len(lidar_rows)}",
    ]
    body = [ln for ln in refined_lines if ln.strip() and not ln.strip().startswith("#")]
    out_txt.write_text("\n".join(header + body + lidar_rows) + "\n")
    return n_refined, len(lidar_rows)
