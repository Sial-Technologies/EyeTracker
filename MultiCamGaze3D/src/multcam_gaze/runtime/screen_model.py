"""Physical screen plane in WORLD coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multcam_gaze.core.transform import Transform
from multcam_gaze.geometry.plane import Plane3D


@dataclass(frozen=True)
class ScreenModel:
    """Screen: X right, Y up, Z into screen (viewer at negative Z side)."""

    width_mm: float
    height_mm: float
    world_from_screen: Transform

    def plane_in_world(self) -> Plane3D:
        """Screen plane z=0 in screen frame."""
        origin = self.world_from_screen.apply_point(np.zeros(3))
        normal = self.world_from_screen.apply_direction(np.array([0.0, 0.0, 1.0]))
        return Plane3D.from_point_normal(origin, normal, frame="world")

    def world_point_to_screen_mm(self, point_world: np.ndarray) -> np.ndarray:
        screen_from_world = self.world_from_screen.inverse()
        p = screen_from_world.apply_point(point_world)
        return p[:2]

    def screen_mm_to_pixels(
        self,
        x_mm: float,
        y_mm: float,
        pixel_width: int,
        pixel_height: int,
    ) -> tuple[float, float]:
        u = (x_mm / self.width_mm + 0.5) * pixel_width
        v = (0.5 - y_mm / self.height_mm) * pixel_height
        return float(u), float(v)
