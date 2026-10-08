"""Project path helpers."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """MultiCamGaze3D repo root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    return project_root() / "config"


def default_calib_dir() -> Path:
    return project_root() / "calib"
