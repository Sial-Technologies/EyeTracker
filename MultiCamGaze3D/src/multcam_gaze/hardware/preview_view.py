"""Preview zoom/pan (pure OpenCV warps)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.types import PREVIEW_ROLES


def apply_zoom_pan(
    canvas: NDArray[np.uint8],
    zoom_level: float,
    pan_x: float,
    pan_y: float,
) -> NDArray[np.uint8]:
    if zoom_level == 1.0 and pan_x == 0 and pan_y == 0:
        return canvas
    h, w = canvas.shape[:2]
    center_x, center_y = w / 2, h / 2
    m = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_level)
    m[0, 2] += pan_x
    m[1, 2] += pan_y
    return cv2.warpAffine(
        canvas,
        m,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def apply_camera_display(
    frame: NDArray[np.uint8],
    flip_vertical: bool,
    flip_horizontal: bool,
    view: dict | None,
) -> NDArray[np.uint8]:
    """Operator preview: flip/mirror first, then zoom/pan (matches tracking crop)."""
    out = frame
    if flip_vertical:
        out = cv2.flip(out, 0)
    if flip_horizontal:
        out = cv2.flip(out, 1)
    v = normalize_preview_view(view if isinstance(view, dict) else None)
    return apply_zoom_pan(out, v["zoom"], v["pan_x"], v["pan_y"])


def prepare_eye_tracking_frame(
    frame: NDArray[np.uint8],
    flip_vertical: bool,
    flip_horizontal: bool,
    view: dict | None,
    zoom_affects_tracking: bool,
) -> tuple[NDArray[np.uint8], bool, bool]:
    """Frame + flip flags for eye_tracker.process_frame.

    When zoom affects tracking, apply flip/mirror *before* zoom/pan so pan offsets
    match the upright (post-flip) image the operator tuned in preview. Flip flags
    returned as False because they are already applied.
    """
    if not zoom_affects_tracking:
        return frame, flip_vertical, flip_horizontal
    return (
        apply_camera_display(frame, flip_vertical, flip_horizontal, view),
        False,
        False,
    )


def effective_tracking_fov_y_deg(
    zoom_affects_tracking: bool,
    view: dict | None,
    *,
    base_fov_y_deg: float = 80.0,
) -> float:
    """Legacy FOV for centered digital zoom when sensor remapping is unavailable.

    Prefer remapping tracker pixels to sensor space + calibrated IR ``K`` (see
    ``tracking.pixel_remap``). ``FOV/z`` is only exact for pan≈0; with pan the
    principal point shifts and this approximation biases gaze angles.
    """
    base = float(base_fov_y_deg)
    if not zoom_affects_tracking:
        return base
    zoom = float(normalize_preview_view(view)["zoom"])
    return base / max(zoom, 1e-6)


def compose_eye_preview_panel(
    raw_frame: NDArray[np.uint8],
    overlay: NDArray[np.uint8] | None,
    flip_vertical: bool,
    flip_horizontal: bool,
    view: dict | None,
    zoom_affects_tracking: bool,
) -> NDArray[np.uint8]:
    """IR panel matching multcam-preview: configured flip/zoom, plus tracker overlay when available.

    When zoom affects tracking, the overlay was drawn on the zoomed frame — use it.
    Otherwise zoom is display-only: start from overlay (already flipped by tracker) or
    the configured display transform of the raw frame.
    """
    configured = apply_camera_display(
        raw_frame, flip_vertical, flip_horizontal, view
    )
    if overlay is None:
        return configured
    if zoom_affects_tracking:
        return overlay
    v = normalize_preview_view(view if isinstance(view, dict) else None)
    return apply_zoom_pan(overlay, v["zoom"], v["pan_x"], v["pan_y"])


def make_status_panel(
    width: int,
    height: int,
    lines: list[str],
    border: tuple[int, int, int] = (80, 80, 80),
) -> NDArray[np.uint8]:
    """Dark placeholder tile with centered status text (avoids a silent black panel)."""
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border, 2)
    cy = height // 2 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = max(8, (width - tw) // 2)
        ty = cy + i * 28
        cv2.putText(panel, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(panel, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return panel


def normalize_preview_view(view: dict | None) -> dict:
    if not isinstance(view, dict):
        return {"zoom": 1.0, "pan_x": 0, "pan_y": 0}
    try:
        zoom = float(view.get("zoom", 1.0))
    except (TypeError, ValueError):
        zoom = 1.0
    try:
        pan_x = int(round(float(view.get("pan_x", 0))))
        pan_y = int(round(float(view.get("pan_y", 0))))
    except (TypeError, ValueError):
        pan_x, pan_y = 0, 0
    return {
        "zoom": max(0.5, min(5.0, round(zoom, 1))),
        "pan_x": pan_x,
        "pan_y": pan_y,
    }


def load_camera_setup(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_camera_setup(path: Path, setup: dict) -> None:
    """Persist role entries (device_id, index, flip, mirror, view, …)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")


def preview_views_from_setup(setup: dict) -> dict:
    views = {role: {"zoom": 1.0, "pan_x": 0, "pan_y": 0} for role in PREVIEW_ROLES}
    for role in PREVIEW_ROLES:
        entry = setup.get(role)
        if isinstance(entry, dict) and isinstance(entry.get("view"), dict):
            views[role] = normalize_preview_view(entry["view"])
    return views


def flip_mirror_from_setup(setup: dict) -> dict[str, dict[str, bool]]:
    flags = {role: {"flip": False, "mirror": False} for role in PREVIEW_ROLES}
    for role in PREVIEW_ROLES:
        entry = setup.get(role)
        if isinstance(entry, dict):
            flags[role] = {
                "flip": bool(entry.get("flip", False)),
                "mirror": bool(entry.get("mirror", False)),
            }
    return flags


def zoom_affects_tracking_from_setup(setup: dict) -> dict[str, bool]:
    flags = {role: False for role in PREVIEW_ROLES}
    for role in PREVIEW_ROLES:
        entry = setup.get(role)
        if isinstance(entry, dict):
            flags[role] = bool(entry.get("zoom_affects_tracking", False))
    return flags


def eye_center_front_mm_from_setup(
    setup: dict,
    preview_role: str,
) -> NDArray[np.float64]:
    """Eyeball center in front OpenCV frame (mm). Default origin = front camera.

    Convention: X right, Y down, Z forward (toward screen). CAD / mount estimate
    of the eyeball relative to the front camera on the same rigid headset.
    """
    entry = setup.get(preview_role)
    if not isinstance(entry, dict):
        return np.zeros(3, dtype=np.float64)
    raw = entry.get("eye_center_front_mm")
    if raw is None:
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        return np.zeros(3, dtype=np.float64)
    return arr


def eye_center_ir_px_from_setup(
    setup: dict,
    preview_role: str,
) -> tuple[int, int] | None:
    """Locked Orlosky 2D eyeball center in the IR tracking buffer, if present."""
    entry = setup.get(preview_role)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("eye_center_ir_px")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 2:
        return None
    return int(round(float(arr[0]))), int(round(float(arr[1])))


def set_eye_center_ir_px(
    setup: dict,
    preview_role: str,
    xy: tuple[int, int] | None,
) -> None:
    """Write or clear ``eye_center_ir_px`` on an existing role entry."""
    entry = setup.get(preview_role)
    if not isinstance(entry, dict):
        return
    if xy is None:
        entry.pop("eye_center_ir_px", None)
        return
    entry["eye_center_ir_px"] = [int(xy[0]), int(xy[1])]


def merge_setup_display(
    setup: dict,
    views: dict[str, dict],
    flips: dict[str, dict[str, bool]],
) -> dict:
    """Update flip/mirror/view on existing role entries; leave device_id/index alone."""
    out: dict = {}
    for role in PREVIEW_ROLES:
        entry = setup.get(role)
        if not isinstance(entry, dict):
            continue
        updated = dict(entry)
        role_flips = flips.get(role, {})
        updated["flip"] = bool(role_flips.get("flip", entry.get("flip", False)))
        updated["mirror"] = bool(role_flips.get("mirror", entry.get("mirror", False)))
        updated["view"] = normalize_preview_view(views.get(role))
        out[role] = updated
    # Preserve any non-role keys if present.
    for key, value in setup.items():
        if key not in out:
            out[key] = value
    return out


def role_camera_index(setup: dict, preview_role: str) -> tuple[int | None, str | None]:
    entry = setup.get(preview_role)
    if not isinstance(entry, dict):
        return None, None
    idx = entry.get("index")
    device_id = entry.get("device_id")
    try:
        idx_int = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx_int = None
    return idx_int, str(device_id) if device_id else None
