"""COLMAP refinement stage: SIFT features → sequential matching →
incremental SfM (full mapping) → bundle adjustment.

The user-supplied PINHOLE intrinsics are seeded into the COLMAP DB at feature
extraction time (CameraMode.SINGLE + `camera_params=<avg fx,fy,cx,cy>`) and
held fixed during BA (`ba_refine_focal_length=False`, etc.). Poses + 3D points
are recovered freshly by incremental_mapping; the smartphone pose priors are
only used to set initial intrinsics, not as fixed poses (pycolmap 4.x's
triangulate_points pathway requires DB-generated frame structures that
text-read reconstructions don't match cleanly).

Uses modern pycolmap 4.x. The `scene_manager_compat` adapter is registered
elsewhere for gsplat downstream; this module itself does not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pycolmap

from intrinsics_utils import AveragedIntrinsics


@dataclass
class ColmapConfig:
    max_num_features: int = 8192
    use_gpu: bool = True
    seq_overlap: int = 10
    seq_quadratic_overlap: bool = True
    loop_detection: bool = False
    refine_focal_length: bool = False
    refine_principal_point: bool = False
    refine_extra_params: bool = False
    refine_extrinsics: bool = True
    refine_points3D: bool = True
    ba_max_iterations: int = 100


@dataclass
class RefinementSummary:
    n_images: int
    n_features_db: int
    n_matches_db: int
    n_points_triangulated: int
    n_points_after_ba: int
    ba_initial_cost: float
    ba_final_cost: float
    intrinsics_used: dict
    ba_options: dict


def _count_db_rows(db_path: Path, table: str) -> int:
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()
    return int(n)


def _patch_db_camera(db_path: Path, intr: AveragedIntrinsics) -> None:
    """The image_reader's SINGLE-camera mode created a camera row from the
    options we passed. Overwrite its params just in case feature_extractor
    rounded or recomputed anything."""
    import struct
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        # PINHOLE model id in COLMAP = 1. params = [fx, fy, cx, cy] as float64.
        params_blob = struct.pack("<4d", intr.fx, intr.fy, intr.cx, intr.cy)
        con.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=1",
            (1, intr.width, intr.height, params_blob),
        )
        con.commit()
    finally:
        con.close()


def refine(
    image_dir: Path,
    sparse_in: Path,
    sparse_out: Path,
    database_path: Path,
    intrinsics: AveragedIntrinsics,
    cfg: ColmapConfig,
    force: bool = False,
) -> RefinementSummary:
    image_dir = Path(image_dir).resolve()
    sparse_in = Path(sparse_in).resolve()
    sparse_out = Path(sparse_out).resolve()
    database_path = Path(database_path).resolve()
    sparse_out.mkdir(parents=True, exist_ok=True)

    if force and database_path.exists():
        database_path.unlink()
    if not database_path.exists():
        # extract_features creates the DB if absent.
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "PINHOLE"
        reader_options.camera_params = f"{intrinsics.fx},{intrinsics.fy},{intrinsics.cx},{intrinsics.cy}"

        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.use_gpu = cfg.use_gpu
        extraction_options.sift.max_num_features = cfg.max_num_features

        print(f"[colmap] extract_features: {image_dir}")
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )
        _patch_db_camera(database_path, intrinsics)

        print("[colmap] match_sequential")
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = cfg.seq_overlap
        pairing_options.quadratic_overlap = cfg.seq_quadratic_overlap
        pairing_options.loop_detection = cfg.loop_detection
        pycolmap.match_sequential(
            database_path=database_path,
            pairing_options=pairing_options,
        )

    n_features_db = _count_db_rows(database_path, "keypoints")
    n_matches_db = _count_db_rows(database_path, "matches")
    print(f"[colmap] DB has {n_features_db} keypoints, {n_matches_db} matched pairs")

    # Run incremental_mapping (full SfM). BA is run repeatedly inside the
    # pipeline; we control which intrinsics it can touch via the pipeline opts.
    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    pipeline_opts.ba_refine_focal_length = cfg.refine_focal_length
    pipeline_opts.ba_refine_principal_point = cfg.refine_principal_point
    pipeline_opts.ba_refine_extra_params = cfg.refine_extra_params
    pipeline_opts.ba_refine_sensor_from_rig = False  # rigs are trivial here
    pipeline_opts.ba_global_max_num_iterations = cfg.ba_max_iterations
    pipeline_opts.ba_local_max_num_iterations = min(cfg.ba_max_iterations, 25)
    # Single-camera + fixed intrinsics: bake in our averaged values.
    pipeline_opts.constant_cameras = {1}

    sparse_intermediate = sparse_out.parent / f"{sparse_out.name}_mapping"
    sparse_intermediate.mkdir(parents=True, exist_ok=True)

    print("[colmap] incremental_mapping (full SfM, intrinsics fixed)")
    recons = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_intermediate,
        options=pipeline_opts,
    )
    if not recons:
        raise RuntimeError("incremental_mapping returned no reconstructions")
    # Pick the largest reconstruction by number of registered images.
    best_id, recon = max(recons.items(), key=lambda kv: kv[1].num_reg_images())
    print(
        f"[colmap] picked reconstruction #{best_id}: "
        f"{recon.num_reg_images()} registered images, {len(recon.points3D)} points"
    )

    # Final BA with our explicit BA options (sanity tightening).
    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.refine_focal_length = cfg.refine_focal_length
    ba_opts.refine_principal_point = cfg.refine_principal_point
    ba_opts.refine_extra_params = cfg.refine_extra_params
    ba_opts.refine_rig_from_world = cfg.refine_extrinsics
    ba_opts.refine_points3D = cfg.refine_points3D
    ba_opts.ceres.solver_options.max_num_iterations = cfg.ba_max_iterations

    print(
        f"[colmap] final bundle_adjustment: refine_focal={cfg.refine_focal_length} "
        f"refine_principal={cfg.refine_principal_point} "
        f"refine_extrinsics={cfg.refine_extrinsics} "
        f"refine_points3D={cfg.refine_points3D} "
        f"max_iter={cfg.ba_max_iterations}"
    )
    pycolmap.bundle_adjustment(reconstruction=recon, options=ba_opts)
    n_points_triangulated = len(recon.points3D)
    n_points_after_ba = len(recon.points3D)

    recon.write_text(str(sparse_out))
    n_images = recon.num_reg_images()
    print(f"[colmap] wrote refined reconstruction to {sparse_out}")

    return RefinementSummary(
        n_images=n_images,
        n_features_db=n_features_db,
        n_matches_db=n_matches_db,
        n_points_triangulated=n_points_triangulated,
        n_points_after_ba=n_points_after_ba,
        ba_initial_cost=float("nan"),
        ba_final_cost=float("nan"),
        intrinsics_used=asdict(intrinsics),
        ba_options={
            "refine_focal_length": cfg.refine_focal_length,
            "refine_principal_point": cfg.refine_principal_point,
            "refine_extra_params": cfg.refine_extra_params,
            "refine_extrinsics": cfg.refine_extrinsics,
            "refine_points3D": cfg.refine_points3D,
            "max_iterations": cfg.ba_max_iterations,
        },
    )
