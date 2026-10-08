"""Homogeneous 4x4 transforms (no OpenCV dependency)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Transform:
    """Rigid transform T_parent_child: maps points from child frame to parent frame.

    P_parent = T_parent_child @ P_child (homogeneous 4-vectors).
    Directions: D_parent = R_parent_child @ D_child (no translation).
    """

    matrix: NDArray[np.float64]
    parent_frame: str
    child_frame: str

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Transform matrix must be 4x4, got {m.shape}")
        object.__setattr__(self, "matrix", m)

    @classmethod
    def identity(cls, frame: str = "same") -> Transform:
        return cls(np.eye(4), frame, frame)

    @classmethod
    def from_rotation_translation(
        cls,
        rotation: NDArray[np.float64],
        translation: NDArray[np.float64],
        parent_frame: str,
        child_frame: str,
    ) -> Transform:
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r
        m[:3, 3] = t
        return cls(m, parent_frame, child_frame)

    @property
    def rotation(self) -> NDArray[np.float64]:
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> NDArray[np.float64]:
        return self.matrix[:3, 3].copy()

    def inverse(self) -> Transform:
        r = self.rotation
        t = self.translation
        r_inv = r.T
        t_inv = -r_inv @ t
        return Transform.from_rotation_translation(
            r_inv, t_inv, self.child_frame, self.parent_frame
        )

    def compose(self, other: Transform) -> Transform:
        """Return T_parent_grandchild = T_parent_child @ T_child_grandchild."""
        if self.child_frame != other.parent_frame:
            raise ValueError(
                f"Frame mismatch: {self.child_frame} != {other.parent_frame} "
                f"for {self.parent_frame}_from_{self.child_frame} @ "
                f"{other.parent_frame}_from_{other.child_frame}"
            )
        return Transform(
            self.matrix @ other.matrix,
            self.parent_frame,
            other.child_frame,
        )

    def apply_point(self, point_child: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(point_child, dtype=np.float64).reshape(3)
        ph = np.ones(4, dtype=np.float64)
        ph[:3] = p
        out = self.matrix @ ph
        return out[:3]

    def apply_direction(self, direction_child: NDArray[np.float64]) -> NDArray[np.float64]:
        d = np.asarray(direction_child, dtype=np.float64).reshape(3)
        out = self.rotation @ d
        n = np.linalg.norm(out)
        if n > 1e-12:
            out = out / n
        return out

    def to_list(self) -> list[list[float]]:
        return self.matrix.tolist()

    @classmethod
    def from_list(
        cls,
        data: list[list[float]],
        parent_frame: str,
        child_frame: str,
    ) -> Transform:
        return cls(np.array(data, dtype=np.float64), parent_frame, child_frame)
