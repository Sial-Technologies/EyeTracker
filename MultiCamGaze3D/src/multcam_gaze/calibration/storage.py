"""Atomic calibration persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from multcam_gaze import __version__
from multcam_gaze.calibration.schema import CALIBRATION_SCHEMA_VERSION, empty_device_calibration
from multcam_gaze.core.transform import Transform
from multcam_gaze.exceptions import CalibrationSchemaError


@dataclass(frozen=True)
class LeftHeatmapCalibration:
    """Persisted left-eye ray map used by the live heatmap."""

    front_from_left: Transform
    direction_scale: float
    scale_yaw: float
    scale_pitch: float
    gaze_error_mm: float | None = None
    path: Path | None = None


class CalibrationStore:
    def __init__(self, calib_dir: Path) -> None:
        self.calib_dir = calib_dir
        self.device_path = calib_dir / "device_calibration.json"

    def load(self) -> dict:
        if not self.device_path.is_file():
            return empty_device_calibration()
        data = json.loads(self.device_path.read_text(encoding="utf-8"))
        if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationSchemaError(
                f"Unsupported schema_version {data.get('schema_version')}"
            )
        return data

    def load_from_path(self, path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise CalibrationSchemaError(
                f"Unsupported schema_version {data.get('schema_version')}"
            )
        return data

    def load_left_heatmap(self, path: Path | None = None) -> LeftHeatmapCalibration:
        """Load ``front_from_left_eye`` + yaw/pitch scales for heatmap-only mode."""
        cal_path = path if path is not None else self.device_path
        data = self.load_from_path(cal_path)
        ext = data.get("extrinsics") if isinstance(data.get("extrinsics"), dict) else {}
        raw = ext.get("front_from_left_eye")
        if not isinstance(raw, dict):
            raise CalibrationSchemaError(
                f"{cal_path} has no extrinsics.front_from_left_eye — run a look-at solve first"
            )
        transform = self.transform_from_json(raw)
        left = {}
        gaze = data.get("gaze_calibration")
        if isinstance(gaze, dict) and isinstance(gaze.get("left"), dict):
            left = gaze["left"]
        quality = data.get("quality") if isinstance(data.get("quality"), dict) else {}
        err = quality.get("gaze_error_mm")
        return LeftHeatmapCalibration(
            front_from_left=transform,
            direction_scale=float(left.get("direction_scale", 1.0)),
            scale_yaw=float(left.get("scale_yaw", 1.0)),
            scale_pitch=float(left.get("scale_pitch", 1.0)),
            gaze_error_mm=float(err) if err is not None else None,
            path=cal_path,
        )

    def save(self, data: dict) -> None:
        data = dict(data)
        data["schema_version"] = CALIBRATION_SCHEMA_VERSION
        data["updated_at"] = datetime.now(UTC).isoformat()
        data["tool_version"] = __version__
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.device_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.device_path)

    @staticmethod
    def transform_to_json(t: Transform) -> dict:
        return {
            "parent_frame": t.parent_frame,
            "child_frame": t.child_frame,
            "matrix": t.to_list(),
        }

    @staticmethod
    def transform_from_json(data: dict) -> Transform:
        return Transform.from_list(
            data["matrix"],
            data["parent_frame"],
            data["child_frame"],
        )
