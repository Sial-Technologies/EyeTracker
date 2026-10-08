"""World gaze rays and screen intersection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multcam_gaze.core.transform import Transform
from multcam_gaze.geometry.plane import intersect_ray_plane
from multcam_gaze.geometry.ray import Ray3D
from multcam_gaze.runtime.screen_model import ScreenModel
from multcam_gaze.types import GazeRayInCameraFrame


@dataclass(frozen=True)
class GazeScreenHit:
    valid: bool
    point_world: np.ndarray | None
    screen_mm: tuple[float, float] | None
    pixels: tuple[float, float] | None
    parallel: bool = False
    on_screen: bool = False


def gaze_to_world_ray(
    gaze: GazeRayInCameraFrame,
    world_from_camera: Transform,
) -> Ray3D:
    c_world = world_from_camera.apply_point(gaze.origin)
    d_world = world_from_camera.apply_direction(gaze.direction)
    return Ray3D(c_world, d_world, frame="world")


def intersect_gaze_with_screen(
    ray_world: Ray3D,
    screen: ScreenModel,
    pixel_width: int,
    pixel_height: int,
    parallel_epsilon: float = 1e-6,
) -> GazeScreenHit:
    plane = screen.plane_in_world()
    hit = intersect_ray_plane(ray_world, plane, parallel_epsilon)
    if hit is None:
        return GazeScreenHit(False, None, None, None, parallel=True, on_screen=False)
    _, point = hit
    xy = screen.world_point_to_screen_mm(point)
    px = screen.screen_mm_to_pixels(xy[0], xy[1], pixel_width, pixel_height)
    u, v = px
    on_screen = 0.0 <= u < float(pixel_width) and 0.0 <= v < float(pixel_height)
    return GazeScreenHit(
        True,
        point,
        (float(xy[0]), float(xy[1])),
        px,
        parallel=False,
        on_screen=on_screen,
    )
