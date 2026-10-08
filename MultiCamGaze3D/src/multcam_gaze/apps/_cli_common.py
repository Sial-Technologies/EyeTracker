"""Shared CLI arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from multcam_gaze.paths import default_calib_dir, default_config_dir


def add_standard_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory (default: MultiCamGaze3D/config)",
    )
    parser.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Calibration output directory (default: MultiCamGaze3D/calib)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    config_dir = args.config_dir or default_config_dir()
    calib_dir = args.calib_dir or default_calib_dir()
    return config_dir, calib_dir
