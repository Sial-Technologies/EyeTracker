"""Screen-space gaze heatmap accumulation / overlay."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def accumulate_heatmap(
    heat: NDArray[np.float32],
    u: float,
    v: float,
    sigma: float = 28.0,
    gain: float = 1.0,
) -> None:
    h, w = heat.shape
    x0 = max(0, int(u - 3 * sigma))
    x1 = min(w, int(u + 3 * sigma) + 1)
    y0 = max(0, int(v - 3 * sigma))
    y1 = min(h, int(v + 3 * sigma) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    blob = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2.0 * sigma * sigma))
    heat[y0:y1, x0:x1] += (gain * blob).astype(np.float32)


def overlay_heatmap(
    canvas: NDArray[np.uint8],
    heat: NDArray[np.float32],
) -> NDArray[np.uint8]:
    if float(np.max(heat)) < 1e-6:
        return canvas
    norm = np.clip(heat / (np.max(heat) + 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(canvas, 0.55, colored, 0.45, 0)
