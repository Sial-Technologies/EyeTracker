"""3D ray representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Ray3D:
    """Ray P(lambda) = origin + lambda * direction (direction need not be unit)."""

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]
    frame: str = "world"

    def point_at(self, lam: float) -> NDArray[np.float64]:
        return self.origin + lam * self.direction

    def normalized_direction(self) -> NDArray[np.float64]:
        d = np.asarray(self.direction, dtype=np.float64).reshape(3)
        n = np.linalg.norm(d)
        if n < 1e-12:
            return d
        return d / n
