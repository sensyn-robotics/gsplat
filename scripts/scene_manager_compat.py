"""Adapter that wraps `pycolmap.Reconstruction` and re-exposes its data under
the attribute names used by gsplat's `examples/datasets/colmap.py` parser.

Register at startup with:

    import pycolmap
    from scripts.scene_manager_compat import SceneManagerAdapter
    pycolmap.SceneManager = SceneManagerAdapter

so gsplat's `from pycolmap import SceneManager` keeps working with modern
pycolmap (4.x), which dropped the SceneManager class. The official
`pycolmap.Reconstruction(path)` is doing the actual file I/O here -- we only
rename attributes and synthesize the inverted lookups gsplat expects.
"""

from __future__ import annotations

import numpy as np
import pycolmap

_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
}


class _CameraView:
    """One per pycolmap camera. Exposes the attribute names gsplat reads."""

    def __init__(self, cam: pycolmap.Camera):
        self._cam = cam
        params = list(cam.params)
        model_name = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
        self.camera_type = _MODEL_NAME_TO_ID.get(model_name, model_name)
        self.width = cam.width
        self.height = cam.height
        self.fx = cam.focal_length_x
        self.fy = cam.focal_length_y
        self.cx = cam.principal_point_x
        self.cy = cam.principal_point_y
        self.k1 = self.k2 = self.k3 = self.k4 = 0.0
        self.k5 = self.k6 = 0.0
        self.p1 = self.p2 = 0.0
        if model_name == "SIMPLE_RADIAL":
            self.k1 = params[3]
        elif model_name == "RADIAL":
            self.k1, self.k2 = params[3], params[4]
        elif model_name == "OPENCV":
            self.k1, self.k2, self.p1, self.p2 = params[4], params[5], params[6], params[7]
        elif model_name == "OPENCV_FISHEYE":
            self.k1, self.k2, self.k3, self.k4 = params[4], params[5], params[6], params[7]


class _ImageView:
    """One per pycolmap image. `R()` and `tvec` expose the world-to-camera pose
    in the same shape gsplat uses (3x3 rotation matrix and (3,) translation)."""

    def __init__(self, img: pycolmap.Image):
        self._img = img
        self.name = img.name
        self.camera_id = img.camera_id
        cw = img.cam_from_world()
        self._R = np.asarray(cw.rotation.matrix(), dtype=np.float64)
        self.tvec = np.asarray(cw.translation, dtype=np.float64)
        self._points2D = img.points2D

    def R(self) -> np.ndarray:
        return self._R

    @property
    def points2D(self):
        return self._points2D


class SceneManagerAdapter:
    """Drop-in replacement for the old `pycolmap.SceneManager` class. Backed by
    `pycolmap.Reconstruction`."""

    def __init__(self, colmap_folder: str, image_path: str | None = None):
        self._recon = pycolmap.Reconstruction(str(colmap_folder).rstrip("/"))
        self.cameras = {cid: _CameraView(c) for cid, c in self._recon.cameras.items()}
        self.images = {iid: _ImageView(i) for iid, i in self._recon.images.items()}
        self.name_to_image_id = {iv.name: iid for iid, iv in self.images.items()}

        ids = sorted(self._recon.points3D.keys())
        self.point3D_ids = np.asarray(ids, dtype=np.uint64)
        if ids:
            self.points3D = np.stack([self._recon.points3D[i].xyz for i in ids]).astype(np.float64)
            self.point3D_colors = np.stack([self._recon.points3D[i].color for i in ids]).astype(np.uint8)
            self.point3D_errors = np.asarray([self._recon.points3D[i].error for i in ids], dtype=np.float64)
        else:
            self.points3D = np.empty((0, 3), dtype=np.float64)
            self.point3D_colors = np.empty((0, 3), dtype=np.uint8)
            self.point3D_errors = np.empty((0,), dtype=np.float64)

        self.point3D_id_to_point3D_idx = {pid: idx for idx, pid in enumerate(ids)}
        self.point3D_id_to_images: dict[int, list[tuple[int, int]]] = {pid: [] for pid in ids}
        for iid, img in self._recon.images.items():
            for p2d_idx, p2d in enumerate(img.points2D):
                pid = int(p2d.point3D_id)
                if pid in self.point3D_id_to_images:
                    self.point3D_id_to_images[pid].append((iid, p2d_idx))

        self.last_image_id = max(self.images.keys()) if self.images else 0
        self.last_camera_id = max(self.cameras.keys()) if self.cameras else 0

    # gsplat calls these but `pycolmap.Reconstruction` loads everything in its
    # constructor; nothing more to do.
    def load_cameras(self):
        pass

    def load_images(self):
        pass

    def load_points3D(self):
        pass


def install():
    """Register the adapter as `pycolmap.SceneManager`. Idempotent."""
    pycolmap.SceneManager = SceneManagerAdapter
