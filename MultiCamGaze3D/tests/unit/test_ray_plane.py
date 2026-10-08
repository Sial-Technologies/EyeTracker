import numpy as np

from multcam_gaze.geometry.plane import Plane3D, intersect_ray_plane
from multcam_gaze.geometry.ray import Ray3D


def test_ray_plane_intersection():
    plane = Plane3D.from_point_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    ray = Ray3D(np.array([0, 0, 10]), np.array([0, 0, -1]))
    hit = intersect_ray_plane(ray, plane)
    assert hit is not None
    lam, pt = hit
    assert lam == 10.0
    np.testing.assert_allclose(pt, [0, 0, 0])


def test_parallel_ray_rejected():
    plane = Plane3D.from_point_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    ray = Ray3D(np.array([0, 0, 1]), np.array([1, 0, 0]))
    assert intersect_ray_plane(ray, plane) is None


def test_behind_ray_rejected():
    plane = Plane3D.from_point_normal(np.array([0, 0, 0]), np.array([0, 0, 1]))
    ray = Ray3D(np.array([0, 0, 10]), np.array([0, 0, 1]))
    assert intersect_ray_plane(ray, plane) is None
