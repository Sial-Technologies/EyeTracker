"""Offline grid-search for left ``eye_center_front_mm`` from look-at dumps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from multcam_gaze.apps._cli_common import add_standard_paths, resolve_paths
from multcam_gaze.calibration.ray_extrinsic import (
    RayExtrinsicSample,
    refine_eye_origin_front,
    solve_front_from_eye,
)
from multcam_gaze.hardware.preview_view import (
    eye_center_front_mm_from_setup,
    load_camera_setup,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Refine left eye_center_front_mm from one or more --dump-samples "
            ".npz files (near + far grids recommended)."
        )
    )
    p.add_argument(
        "dumps",
        type=Path,
        nargs="+",
        help="One or more left_samples*.npz dumps",
    )
    p.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Seed E as x,y,z mm (default: camera_setup left or dump eye_front)",
    )
    p.add_argument(
        "--half-extent-mm",
        type=float,
        default=40.0,
        help="Coarse search half-range around seed (default 40)",
    )
    p.add_argument(
        "--step-mm",
        type=float,
        default=5.0,
        help="Coarse grid step mm (default 5)",
    )
    p.add_argument(
        "--fine-half-extent-mm",
        type=float,
        default=10.0,
        help="Fine search half-range (default 10)",
    )
    p.add_argument(
        "--fine-step-mm",
        type=float,
        default=2.0,
        help="Fine grid step mm (default 2)",
    )
    p.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Score all dumps as one pool without plane projection "
            "(default: average per-dump miss with plane denoise)."
        ),
    )
    p.add_argument(
        "--write-setup",
        action="store_true",
        help="Write best E into config/camera_setup.json left.eye_center_front_mm",
    )
    p.add_argument(
        "--seed-prior",
        type=float,
        default=None,
        help=(
            "Penalty (mm miss per mm |E-seed|); default from library. "
            "Use 0 to disable (may walk to grid edge)."
        ),
    )
    add_standard_paths(p)
    return p.parse_args(argv)


def _parse_seed(text: str) -> NDArray[np.float64]:
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z seed, got {text!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def load_samples_npz(path: Path) -> tuple[list[RayExtrinsicSample], NDArray[np.float64] | None]:
    data = np.load(path)
    dirs = np.asarray(data["directions"], dtype=np.float64)
    targets = np.asarray(data["targets_front"], dtype=np.float64)
    origins = np.asarray(data["origins"], dtype=np.float64)
    uvs = np.asarray(data["target_uvs"], dtype=np.float64)
    reproj = (
        np.asarray(data["aruco_reproj_px"], dtype=np.float64)
        if "aruco_reproj_px" in data.files
        else None
    )
    markers = (
        np.asarray(data["aruco_markers"], dtype=np.float64)
        if "aruco_markers" in data.files
        else None
    )
    eye_front = (
        np.asarray(data["eye_front"], dtype=np.float64).reshape(3)
        if "eye_front" in data.files
        else None
    )
    samples: list[RayExtrinsicSample] = []
    for i in range(len(dirs)):
        uv = uvs[i]
        uv_t = (float(uv[0]), float(uv[1])) if np.all(np.isfinite(uv)) else None
        rp = (
            float(reproj[i])
            if reproj is not None and np.isfinite(reproj[i])
            else None
        )
        mk = (
            int(markers[i])
            if markers is not None and np.isfinite(markers[i])
            else None
        )
        samples.append(
            RayExtrinsicSample(
                origins[i],
                dirs[i],
                targets[i],
                uv_t,
                aruco_reproj_px=rp,
                aruco_markers=mk,
            )
        )
    return samples, eye_front


def _write_eye_center(setup_path: Path, eye: NDArray[np.float64]) -> None:
    setup = load_camera_setup(setup_path)
    left = setup.get("left")
    if not isinstance(left, dict):
        raise ValueError(f"No left entry in {setup_path}")
    left["eye_center_front_mm"] = [
        float(eye[0]),
        float(eye[1]),
        float(eye[2]),
    ]
    setup_path.write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")


def _mean_full_resolve_mm(
    groups: list[list[RayExtrinsicSample]],
    eye: NDArray[np.float64],
    *,
    project_plane: bool,
) -> tuple[float, float, float, float, int]:
    """Average full solve metrics across groups (per-dump) or one joint pool."""
    coplanar = 80.0 if project_plane else None
    misses: list[float] = []
    degs: list[float] = []
    yaws: list[float] = []
    pitches: list[float] = []
    used = 0
    for g in groups:
        r = solve_front_from_eye(
            g,
            ray_origin_front=eye,
            project_to_consensus_plane=project_plane,
            coplanar_max_dist_mm=coplanar,
        )
        misses.append(float(r.mean_residual_mm))
        degs.append(float(r.mean_residual_deg))
        yaws.append(float(r.scale_yaw))
        pitches.append(float(r.scale_pitch))
        used += int(r.n_samples_used)
    return (
        float(np.mean(misses)),
        float(np.mean(degs)),
        float(np.mean(yaws)),
        float(np.mean(pitches)),
        used,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_dir, _ = resolve_paths(args)
    setup_path = config_dir / "camera_setup.json"

    groups: list[list[RayExtrinsicSample]] = []
    dump_eye: NDArray[np.float64] | None = None
    for path in args.dumps:
        if not path.is_file():
            print(f"Missing dump: {path}", file=sys.stderr)
            return 1
        samples, eye = load_samples_npz(path)
        print(f"Loaded {len(samples)} samples from {path}")
        groups.append(samples)
        if dump_eye is None and eye is not None:
            dump_eye = eye

    n_total = sum(len(g) for g in groups)
    if n_total < 3:
        print("Need at least 3 samples total", file=sys.stderr)
        return 1

    if args.seed is not None:
        seed = _parse_seed(args.seed)
        seed_src = "--seed"
    elif setup_path.is_file():
        setup = load_camera_setup(setup_path)
        seed = eye_center_front_mm_from_setup(setup, "left")
        seed_src = str(setup_path)
    elif dump_eye is not None:
        seed = dump_eye
        seed_src = "dump eye_front"
    else:
        seed = np.zeros(3, dtype=np.float64)
        seed_src = "default [0,0,0]"

    joint = bool(args.joint)
    project_plane = not joint
    coplanar = 80.0 if project_plane else None
    refine_input: list[RayExtrinsicSample] | list[list[RayExtrinsicSample]]
    if joint:
        refine_input = [s for g in groups for s in g]
        score_mode = "joint (no plane)"
    else:
        refine_input = groups
        score_mode = "mean per-dump (plane denoise)"

    print(
        f"Seed E={seed.tolist()} ({seed_src})  "
        f"n={n_total} dumps={len(groups)}  mode={score_mode}"
    )

    prior_kw: dict = {}
    if args.seed_prior is not None:
        prior_kw["seed_prior_per_mm"] = float(args.seed_prior)

    result = refine_eye_origin_front(
        refine_input,
        seed=seed,
        half_extent_mm=float(args.half_extent_mm),
        step_mm=float(args.step_mm),
        fine_half_extent_mm=float(args.fine_half_extent_mm),
        fine_step_mm=float(args.fine_step_mm),
        project_to_consensus_plane=project_plane,
        coplanar_max_dist_mm=coplanar,
        run_full_solve=False,
        **prior_kw,
    )
    best = result.eye_origin_front
    delta = best - result.seed
    delta_norm = float(np.linalg.norm(delta))

    print(
        f"Kabsch search: seed miss={result.seed_residual_mm:.1f} mm -> "
        f"best miss={result.residual_mm:.1f} mm  evals={result.n_evals}"
    )
    print(
        f"Suggested eye_center_front_mm: "
        f"[{best[0]:.1f}, {best[1]:.1f}, {best[2]:.1f}]  "
        f"(delta from seed [{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}] mm)"
    )
    if delta_norm > 0.85 * float(args.half_extent_mm):
        print(
            "WARN: best E is near the search boundary - check CAD seed / "
            "widen --half-extent-mm only if the headset geometry allows it."
        )

    eval_groups = groups if not joint else [[s for g in groups for s in g]]
    seed_mm, seed_deg, seed_yaw, seed_pitch, seed_used = _mean_full_resolve_mm(
        eval_groups, result.seed, project_plane=project_plane
    )
    best_mm, best_deg, best_yaw, best_pitch, best_used = _mean_full_resolve_mm(
        eval_groups, best, project_plane=project_plane
    )
    print(
        f"Full re-solve at seed E: {seed_mm:.1f} mm / {seed_deg:.2f} deg  "
        f"yaw={seed_yaw:.3f} pitch={seed_pitch:.3f} used={seed_used}"
    )
    print(
        f"Full re-solve at best E: {best_mm:.1f} mm / {best_deg:.2f} deg  "
        f"yaw={best_yaw:.3f} pitch={best_pitch:.3f} used={best_used}"
    )

    print("Paste into config/camera_setup.json -> left:")
    print(
        '  "eye_center_front_mm": ['
        f"{best[0]:.1f}, {best[1]:.1f}, {best[2]:.1f}]"
    )
    print("Then restart multcam-left-heatmap and re-solve (or re-capture) at one distance.")

    if args.write_setup:
        if not setup_path.is_file():
            print(f"Cannot --write-setup: missing {setup_path}", file=sys.stderr)
            return 1
        _write_eye_center(setup_path, best)
        print(f"Wrote {setup_path}")

    if best_mm > seed_mm + 1.0:
        print(
            "WARN: full solve at best E is worse than seed - keep CAD seed or "
            "widen/narrow the search grid.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
