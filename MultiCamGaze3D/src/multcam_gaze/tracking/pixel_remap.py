"""Map Orlosky tracker pixels back to raw IR sensor coordinates.

Tracking may see a flipped + digitally zoomed/panned buffer that is then
``crop_to_aspect_ratio``-resized to 640×480. Gaze unprojection must undo that
chain so pan does not shift the principal point and zoom does not require
``FOV/z`` (which is only valid for centered crops).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Matches eye_tracker.crop_to_aspect_ratio default target.
ORLOSKY_WIDTH = 640
ORLOSKY_HEIGHT = 480


@dataclass(frozen=True)
class ZoomPan:
    zoom: float
    pan_x: float
    pan_y: float

    @classmethod
    def from_view(cls, view: dict | None) -> ZoomPan:
        if not isinstance(view, dict):
            return cls(1.0, 0.0, 0.0)
        return cls(
            float(view.get("zoom", 1.0) or 1.0),
            float(view.get("pan_x", 0.0) or 0.0),
            float(view.get("pan_y", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class TrackerPixelMap:
    """Undo crop_to_aspect → zoom/pan → flips to reach calibration sensor space."""

    sensor_width: int
    sensor_height: int
    orlosky_width: int = ORLOSKY_WIDTH
    orlosky_height: int = ORLOSKY_HEIGHT
    zoom_pan: ZoomPan | None = None
    """Zoom/pan already applied to the frame fed into process_frame (pre-crop)."""
    undo_flip_vertical: bool = False
    undo_flip_horizontal: bool = False

    def to_sensor(self, x: float, y: float) -> tuple[float, float]:
        xb, yb = _orlosky_to_pre_crop(
            x,
            y,
            self.sensor_width,
            self.sensor_height,
            self.orlosky_width,
            self.orlosky_height,
        )
        if self.zoom_pan is not None and (
            abs(self.zoom_pan.zoom - 1.0) > 1e-9
            or abs(self.zoom_pan.pan_x) > 1e-9
            or abs(self.zoom_pan.pan_y) > 1e-9
        ):
            xb, yb = invert_zoom_pan_point(
                xb,
                yb,
                self.sensor_width,
                self.sensor_height,
                self.zoom_pan.zoom,
                self.zoom_pan.pan_x,
                self.zoom_pan.pan_y,
            )
        return undo_flips(
            xb,
            yb,
            self.sensor_width,
            self.sensor_height,
            self.undo_flip_vertical,
            self.undo_flip_horizontal,
        )


def zoom_pan_affine(
    width: int,
    height: int,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> NDArray[np.float64]:
    """Same 2×3 affine as ``cv2.getRotationMatrix2D`` scale + pan (matches preview)."""
    cx = width / 2.0
    cy = height / 2.0
    z = float(zoom)
    return np.array(
        [[z, 0.0, cx * (1.0 - z) + float(pan_x)], [0.0, z, cy * (1.0 - z) + float(pan_y)]],
        dtype=np.float64,
    )


def invert_zoom_pan_point(
    x: float,
    y: float,
    width: int,
    height: int,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> tuple[float, float]:
    """Buffer pixel → pre-zoom sensor pixel (optical center preserved under pan)."""
    m = zoom_pan_affine(width, height, zoom, pan_x, pan_y)
    # Square affine: [z 0 tx; 0 z ty] → inverse [1/z 0 -tx/z; 0 1/z -ty/z]
    z = float(m[0, 0])
    if abs(z) < 1e-9:
        return float(x), float(y)
    tx, ty = float(m[0, 2]), float(m[1, 2])
    return (float(x) - tx) / z, (float(y) - ty) / z


def undo_flips(
    x: float,
    y: float,
    width: int,
    height: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> tuple[float, float]:
    xs, ys = float(x), float(y)
    if flip_horizontal:
        xs = (width - 1) - xs
    if flip_vertical:
        ys = (height - 1) - ys
    return xs, ys


def _crop_window(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    """Return (offset_x, offset_y, crop_w, crop_h) matching eye_tracker.crop_to_aspect_ratio."""
    desired_ratio = dst_w / float(dst_h)
    current_ratio = src_w / float(src_h)
    if current_ratio > desired_ratio:
        crop_w = int(desired_ratio * src_h)
        crop_h = src_h
        offset_x = (src_w - crop_w) // 2
        offset_y = 0
    else:
        crop_w = src_w
        crop_h = int(src_w / desired_ratio)
        offset_x = 0
        offset_y = (src_h - crop_h) // 2
    return offset_x, offset_y, crop_w, crop_h


def _orlosky_to_pre_crop(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[float, float]:
    ox, oy, crop_w, crop_h = _crop_window(src_w, src_h, dst_w, dst_h)
    xs = ox + float(x) * (crop_w / float(dst_w))
    ys = oy + float(y) * (crop_h / float(dst_h))
    return xs, ys
