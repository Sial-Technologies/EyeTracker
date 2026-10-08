"""Map Orlosky tracker pixels back to raw IR sensor coordinates.

Tracking always ``crop_to_aspect_ratio``-resizes to 640×480 after optional
mount flips inside ``process_frame``. Gaze unprojection undoes that crop (and
flips) so Phase-0 ``K`` / ``dist`` apply in calibration sensor space.
"""

from __future__ import annotations

from dataclasses import dataclass


# Matches eye_tracker.crop_to_aspect_ratio default target.
ORLOSKY_WIDTH = 640
ORLOSKY_HEIGHT = 480


@dataclass(frozen=True)
class TrackerPixelMap:
    """Undo crop_to_aspect → flips to reach calibration sensor space."""

    sensor_width: int
    sensor_height: int
    orlosky_width: int = ORLOSKY_WIDTH
    orlosky_height: int = ORLOSKY_HEIGHT
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
        return undo_flips(
            xb,
            yb,
            self.sensor_width,
            self.sensor_height,
            self.undo_flip_vertical,
            self.undo_flip_horizontal,
        )


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
