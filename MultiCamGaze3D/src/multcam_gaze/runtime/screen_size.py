"""Physical screen size: diagonal / mm → metric scale for ArUco PnP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multcam_gaze.core.transform import Transform
from multcam_gaze.geometry.plane import Plane3D
from multcam_gaze.runtime.screen_model import ScreenModel

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class ScreenPhysicalSize:
    """Metric panel size used as the ArUco / PnP scale reference."""

    width_mm: float
    height_mm: float
    diagonal_inches: float
    source: str  # e.g. "diagonal 27\" + 16:9 px"

    @property
    def diagonal_mm(self) -> float:
        return float(np.hypot(self.width_mm, self.height_mm))


def size_from_diagonal_inches(
    diagonal_inches: float,
    aspect_width: float,
    aspect_height: float,
    *,
    source: str | None = None,
) -> ScreenPhysicalSize:
    """Convert marketed diagonal (inches) + panel aspect → width/height mm."""
    d_in = float(diagonal_inches)
    if d_in < 5.0 or d_in > 120.0:
        raise ValueError(f"diagonal_inches out of range: {d_in}")
    aw = float(aspect_width)
    ah = float(aspect_height)
    if aw <= 0 or ah <= 0:
        raise ValueError("aspect width/height must be positive")
    diag_mm = d_in * MM_PER_INCH
    # w^2 + h^2 = diag^2, w/h = aw/ah
    scale = diag_mm / float(np.hypot(aw, ah))
    width_mm = scale * aw
    height_mm = scale * ah
    label = source or f'diagonal {d_in:g}" + aspect {aw:g}:{ah:g}'
    return ScreenPhysicalSize(width_mm, height_mm, d_in, label)


def size_from_width_height_mm(
    width_mm: float,
    height_mm: float,
    *,
    source: str = "width_mm/height_mm",
) -> ScreenPhysicalSize:
    w = float(width_mm)
    h = float(height_mm)
    if w < 50.0 or h < 50.0:
        raise ValueError(f"screen mm too small: {w}x{h}")
    diag_in = float(np.hypot(w, h) / MM_PER_INCH)
    return ScreenPhysicalSize(w, h, diag_in, source)


def load_screen_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _unwrap_config_source(source: object | None) -> str:
    """Strip repeated ``config (...)`` wrappers from older save/reload cycles."""
    if not isinstance(source, str) or not source.strip():
        return "screen.json"
    label = source.strip()
    prefix = "config ("
    while label.startswith(prefix) and label.endswith(")"):
        label = label[len(prefix) : -1].strip()
        if not label:
            return "screen.json"
    return label


def save_screen_config(path: Path, size: ScreenPhysicalSize) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagonal_inches": round(float(size.diagonal_inches), 4),
        "width_mm": round(float(size.width_mm), 3),
        "height_mm": round(float(size.height_mm), 3),
        # Persist the unwrapped provenance so reload does not nest ``config (...)``.
        "source": _unwrap_config_source(size.source),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_screen_physical_size(
    *,
    screen_w_px: int,
    screen_h_px: int,
    diagonal_inches: float | None = None,
    width_mm: float | None = None,
    height_mm: float | None = None,
    config: dict | None = None,
) -> ScreenPhysicalSize:
    """Require an explicit metric reference (diagonal or width+height mm).

    Pixel aspect of the fullscreen window is used when only the diagonal is known.
    GDI estimates are never used as the scale source.
    """
    cfg = config or {}
    if width_mm is not None and height_mm is not None:
        return size_from_width_height_mm(
            width_mm, height_mm, source="cli --width-mm/--height-mm"
        )
    if width_mm is not None or height_mm is not None:
        raise ValueError("Provide both --width-mm and --height-mm, or use --diagonal-inches")

    cfg_w = cfg.get("width_mm")
    cfg_h = cfg.get("height_mm")
    if cfg_w is not None and cfg_h is not None:
        inner = _unwrap_config_source(cfg.get("source"))
        return size_from_width_height_mm(
            float(cfg_w), float(cfg_h), source=f"config ({inner})"
        )

    diag = diagonal_inches
    if diag is None and cfg.get("diagonal_inches") is not None:
        diag = float(cfg["diagonal_inches"])
    if diag is None:
        raise ValueError(
            "Metric screen size required for ArUco scale. Pass --diagonal-inches "
            '(e.g. 27) or set config/screen.json, or both --width-mm and --height-mm. '
            "OS/GDI size is not used as ground truth."
        )
    return size_from_diagonal_inches(
        float(diag),
        float(screen_w_px),
        float(screen_h_px),
        source=f'diagonal {float(diag):g}" + {screen_w_px}x{screen_h_px}px aspect',
    )


def camera_to_screen_distance_mm(front_from_screen: Transform) -> float:
    """Perpendicular distance from front-camera origin to the screen plane (mm)."""
    origin = front_from_screen.apply_point(np.zeros(3))
    normal = front_from_screen.apply_direction(np.array([0.0, 0.0, 1.0]))
    plane = Plane3D.from_point_normal(origin, normal, frame=front_from_screen.parent_frame)
    return abs(float(plane.d))


def camera_to_screen_distance_mm_from_model(screen: ScreenModel) -> float:
    return camera_to_screen_distance_mm(screen.world_from_screen)
