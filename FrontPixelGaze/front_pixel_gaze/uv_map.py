"""Affine map from gaze yaw/pitch (OpenCV convention) to front-camera pixels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def direction_to_yaw_pitch(direction: NDArray[np.float64]) -> tuple[float, float]:
    """OpenCV Y-down unit direction -> yaw (right+), pitch (up+)."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return 0.0, 0.0
    d = d / n
    yaw = float(np.arctan2(d[0], d[2]))
    pitch = float(np.arctan2(-d[1], float(np.hypot(d[0], d[2]))))
    return yaw, pitch


def orlosky_to_opencv_direction(
    direction: NDArray[np.float64],
    *,
    vertical_flip_undone: bool = False,
) -> NDArray[np.float64]:
    """Orlosky / OpenGL gaze is Y-up; convert to OpenCV Y-down.

    When the IR mount flip was undone for sensor-space unprojection (MultiCamGaze
    ``pixel_map`` with ``flip: true``), skip the extra Y negate or pitch is crushed.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3).copy()
    if not vertical_flip_undone:
        d[1] = -d[1]
    n = float(np.linalg.norm(d))
    if n > 1e-12:
        d = d / n
    return d


@dataclass
class FrontUvMap:
    """[u, v]^T = A @ [yaw, pitch, 1]^T with A shape (2, 3)."""

    A: NDArray[np.float64]
    n_samples: int
    rms_px: float

    def apply(
        self,
        direction_opencv: NDArray[np.float64],
        *,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        yaw, pitch = direction_to_yaw_pitch(direction_opencv)
        uv = self.A @ np.array([yaw, pitch, 1.0], dtype=np.float64)
        u, v = float(uv[0]), float(uv[1])
        if image_size is not None:
            w, h = image_size
            u = float(np.clip(u, 0.0, w - 1.0))
            v = float(np.clip(v, 0.0, h - 1.0))
        return u, v

    def to_dict(self) -> dict:
        return {
            "A": self.A.tolist(),
            "n_samples": int(self.n_samples),
            "rms_px": float(self.rms_px),
        }

    @classmethod
    def from_dict(cls, data: dict) -> FrontUvMap:
        return cls(
            A=np.asarray(data["A"], dtype=np.float64).reshape(2, 3),
            n_samples=int(data.get("n_samples", 0)),
            rms_px=float(data.get("rms_px", 0.0)),
        )


def fit_front_uv_map(
    directions_opencv: list[NDArray[np.float64]],
    uvs: list[tuple[float, float]],
) -> FrontUvMap:
    if len(directions_opencv) != len(uvs):
        raise ValueError("directions and uvs length mismatch")
    if len(uvs) < 3:
        raise ValueError("need at least 3 samples to fit affine yaw/pitch → UV")

    rows = []
    targets = []
    for d, (u, v) in zip(directions_opencv, uvs):
        yaw, pitch = direction_to_yaw_pitch(d)
        rows.append([yaw, pitch, 1.0])
        targets.append([u, v])
    X = np.asarray(rows, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    # Solve X @ A.T = Y  =>  A.T = lstsq(X, Y)
    At, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A = At.T
    pred = X @ At
    err = pred - Y
    rms = float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2)))
    return FrontUvMap(A=A, n_samples=len(uvs), rms_px=rms)


def save_front_uv_map(path: Path, model: FrontUvMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_front_uv_map(path: Path) -> FrontUvMap | None:
    if not path.is_file():
        return None
    return FrontUvMap.from_dict(json.loads(path.read_text(encoding="utf-8")))
