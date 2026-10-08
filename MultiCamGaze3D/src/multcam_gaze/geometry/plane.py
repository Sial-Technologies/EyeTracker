"""Plane and ray-plane intersection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multcam_gaze.geometry.ray import Ray3D


@dataclass(frozen=True)
class Plane3D:
    """Plane n·P + d = 0 with unit normal n."""

    normal: NDArray[np.float64]
    d: float
    frame: str = "world"

    @classmethod
    def from_point_normal(
        cls,
        point: NDArray[np.float64],
        normal: NDArray[np.float64],
        frame: str = "world",
    ) -> Plane3D:
        p = np.asarray(point, dtype=np.float64).reshape(3)
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            raise ValueError("Plane normal must be non-zero")
        n = n / nn
        d = -float(np.dot(n, p))
        return cls(n, d, frame)


def intersect_ray_plane(
    ray: Ray3D,
    plane: Plane3D,
    parallel_epsilon: float = 1e-6,
) -> tuple[float, NDArray[np.float64]] | None:
    """Return (lambda, intersection point) or None if parallel / behind origin."""
    n = plane.normal
    denom = float(np.dot(n, ray.direction))
    if abs(denom) < parallel_epsilon:
        return None
    lam = -(float(np.dot(n, ray.origin)) + plane.d) / denom
    if lam < 1e-6:
        return None
    return lam, ray.point_at(lam)


def closest_points_between_rays(
    ray_a: Ray3D,
    ray_b: Ray3D,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Closest points on two infinite lines; returns (point_a, point_b, distance)."""
    p1 = ray_a.origin
    d1 = ray_a.normalized_direction()
    p2 = ray_b.origin
    d2 = ray_b.normalized_direction()
    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        t = 0.0
        s = d / b if abs(b) > 1e-12 else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    pa = p1 + s * d1
    pb = p2 + t * d2
    dist = float(np.linalg.norm(pa - pb))
    return pa, pb, dist
