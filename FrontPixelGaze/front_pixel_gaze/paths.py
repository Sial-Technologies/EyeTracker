"""Project root paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CALIB_DIR = PROJECT_ROOT / "calib"
INTRINSICS_DIR = CALIB_DIR / "intrinsics"
CAMERA_SETUP_PATH = CONFIG_DIR / "camera_setup.json"
FRONT_UV_MAP_PATH = CALIB_DIR / "front_uv_map.json"
