"""Offline numeric checks on dumped left-heatmap look-at samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from multcam_gaze.calibration.ray_extrinsic import (
    RayExtrinsicSample,
    angular_error_deg,
    p_front_z_stats,
    plane_distances_mm,
    point_to_ray_distance,
    scale_gaze_direction,
    solve_front_from_eye,
)

# Stable ArUco geometry: P_front should lie near one plane (tens of mm, not hundreds).
MAX_GOOD_PLANE_ABS_MM = 40.0
MAX_GOOD_Z_SPAN_MM = 80.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze dumped left-heatmap samples (directions vs targets)."
    )
    p.add_argument("dump", type=Path, help="Path to .npz from --dump-samples")
    return p.parse_args(argv)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dump.is_file():
        print(f"Missing dump: {args.dump}", file=sys.stderr)
        return 1

    data = np.load(args.dump)
    dirs = np.asarray(data["directions"], dtype=np.float64)
    targets = np.asarray(data["targets_front"], dtype=np.float64)
    uvs = np.asarray(data["target_uvs"], dtype=np.float64)
    n = len(dirs)
    print(f"Loaded {n} samples from {args.dump}")

    # --- Direction span (tracking usable?) ---
    mean_d = dirs.mean(axis=0)
    mn = float(np.linalg.norm(mean_d))
    mean_d = mean_d / mn if mn > 1e-12 else dirs[0]
    angs = []
    for d in dirs:
        dn = d / max(float(np.linalg.norm(d)), 1e-12)
        c = float(np.clip(np.dot(dn, mean_d), -1.0, 1.0))
        angs.append(float(np.degrees(np.arccos(c))))
    print(
        f"Direction span from mean: max={max(angs):.2f} deg  "
        f"rms={float(np.sqrt(np.mean(np.square(angs)))):.2f} deg"
    )
    if max(angs) < 2.0:
        print(
            "FAIL: gaze directions barely change across targets -> tracking/FOV/flip, "
            "not the extrinsic solver."
        )
    else:
        print("OK: directions vary across the grid (tracking is producing signal).")

    # --- Correlation: dir components vs screen UV ---
    if np.all(np.isfinite(uvs)):
        print("Correlation (dir component vs target pixel):")
        for axis, name in enumerate("xyz"):
            cu = _corr(dirs[:, axis], uvs[:, 0])
            cv = _corr(dirs[:, axis], uvs[:, 1])
            print(f"  d{name}-u={cu:+.3f}  d{name}-v={cv:+.3f}")
        # Healthy: some axis tracks u, another tracks v with |corr| > ~0.5
        best_u = max(abs(_corr(dirs[:, i], uvs[:, 0])) for i in range(3))
        best_v = max(abs(_corr(dirs[:, i], uvs[:, 1])) for i in range(3))
        if best_u < 0.4 or best_v < 0.4:
            print(
                "WARN: weak dir-UV correlation - look-at targets and gaze may be "
                "uncoupled (wrong pupil, flips, or screen pose)."
            )
        else:
            print("OK: at least one dir axis correlates with u and with v.")

    # --- Build samples (with ArUco quality when present) ---
    origins = np.asarray(data["origins"], dtype=np.float64)
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
    samples: list[RayExtrinsicSample] = []
    for i in range(n):
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

    # --- ArUco / P_front geometry (#1 suspect for depth jumps) ---
    z_mean, z_std, z_span = p_front_z_stats(samples)
    plane_dist = plane_distances_mm(samples)
    abs_plane = np.abs(plane_dist) if len(plane_dist) else np.zeros(0)
    print("ArUco / P_front geometry:")
    print(
        f"  P_front Z: mean={z_mean:.1f} mm  std={z_std:.1f} mm  "
        f"span={z_span:.1f} mm"
    )
    if len(abs_plane):
        print(
            f"  Plane |dist|: mean={float(abs_plane.mean()):.1f} mm  "
            f"max={float(abs_plane.max()):.1f} mm"
        )
    if reproj is not None and np.any(np.isfinite(reproj)):
        valid_r = reproj[np.isfinite(reproj)]
        valid_m = (
            markers[np.isfinite(markers)]
            if markers is not None
            else np.array([], dtype=np.float64)
        )
        print(
            f"  ArUco reproj: mean={float(valid_r.mean()):.2f} px  "
            f"max={float(valid_r.max()):.2f} px"
        )
        if len(valid_m):
            print(
                f"  ArUco markers: min={int(valid_m.min())}  "
                f"mean={float(valid_m.mean()):.1f}"
            )
    else:
        print("  (no aruco_reproj_px in dump - re-capture with current app)")

    pose_suspect = False
    if z_span > MAX_GOOD_Z_SPAN_MM:
        print(
            f"FAIL: P_front Z span {z_span:.0f} mm > {MAX_GOOD_Z_SPAN_MM:.0f} mm - "
            "screen points are not a stable plane in the front frame (ArUco/pose)."
        )
        pose_suspect = True
    elif len(abs_plane) and float(abs_plane.max()) > MAX_GOOD_PLANE_ABS_MM:
        print(
            f"WARN: max plane distance {float(abs_plane.max()):.0f} mm > "
            f"{MAX_GOOD_PLANE_ABS_MM:.0f} mm - hold head still, keep all 4 markers."
        )
        pose_suspect = True
    else:
        print("OK: P_front depths look plane-like (tens of mm, not hundreds).")

    # --- Re-solve and compare Y-flip ablation ---
    result = solve_front_from_eye(samples)
    print(
        f"Re-solve: mean {result.mean_residual_mm:.1f} mm / "
        f"{result.mean_residual_deg:.2f} deg  "
        f"yaw={result.scale_yaw:.3f} pitch={result.scale_pitch:.3f} "
        f"used={result.n_samples_used}"
        + (
            f"  polish={result.mean_residual_px:.1f}px (diag only)"
            if result.mean_residual_px is not None
            else ""
        )
    )
    if result.scale_yaw < 0.5 or result.scale_pitch < 0.5:
        print(
            "HINT: solved yaw/pitch scales << 1 — check left_eye.npz, sensor remap "
            "(zoom/pan), and that pan is not applied without pixel remap."
        )

    flipped = []
    for s in samples:
        d = np.asarray(s.direction, dtype=np.float64).copy()
        d[1] = -d[1]
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        flipped.append(
            RayExtrinsicSample(
                s.origin,
                d,
                s.target_front,
                s.target_uv,
                aruco_reproj_px=s.aruco_reproj_px,
                aruco_markers=s.aruco_markers,
            )
        )
    result_flip = solve_front_from_eye(flipped)
    print(
        f"Re-solve with EXTRA Y flip: mean {result_flip.mean_residual_mm:.1f} mm / "
        f"{result_flip.mean_residual_deg:.2f} deg  "
        f"yaw={result_flip.scale_yaw:.3f} pitch={result_flip.scale_pitch:.3f}"
    )
    if result_flip.mean_residual_mm + 5.0 < result.mean_residual_mm:
        print(
            "HINT: extra Y flip fits better - Orlosky->OpenCV Y conversion may be "
            "wrong or double-applied."
        )
    elif result.mean_residual_mm + 5.0 < result_flip.mean_residual_mm:
        print("OK: current Y convention fits better than an extra flip.")
    else:
        print("INFO: Y-flip ablation inconclusive (similar residuals).")

    # --- Per-point table ---
    print("Per-point miss (current re-solve):")
    eye = result.front_from_eye.translation
    for i, s in enumerate(samples):
        d = result.front_from_eye.apply_direction(
            scale_gaze_direction(
                s.direction,
                result.direction_scale,
                scale_yaw=result.scale_yaw,
                scale_pitch=result.scale_pitch,
            )
        )
        mm = point_to_ray_distance(eye, d, s.target_front)
        deg = angular_error_deg(eye, d, s.target_front)
        uv = s.target_uv
        uv_txt = f"uv=({uv[0]:.0f},{uv[1]:.0f})" if uv is not None else "uv=(?,?)"
        p = np.asarray(s.target_front, dtype=np.float64).reshape(3)
        aruco = ""
        if s.aruco_reproj_px is not None:
            aruco = (
                f"  aruco={s.aruco_reproj_px:.2f}px/{s.aruco_markers}  "
                f"plane={plane_dist[i]:+.1f}mm"
            )
        print(
            f"  [{i + 1:02d}] {uv_txt}  {mm:7.1f} mm  {deg:5.2f} deg  "
            f"Z={p[2]:.0f}mm{aruco}"
        )

    # --- Heatmap right-click debug (optional; written when --dump-samples set) ---
    if "debug_truth_uv" in data.files:
        truth = np.asarray(data["debug_truth_uv"], dtype=np.float64)
        pred = np.asarray(data["debug_pred_uv"], dtype=np.float64)
        du = np.asarray(data["debug_du_px"], dtype=np.float64)
        dv = np.asarray(data["debug_dv_px"], dtype=np.float64)
        err_mm = np.asarray(data["debug_err_mm"], dtype=np.float64)
        n_dbg = len(truth)
        print(f"Heatmap debug clicks: {n_dbg}")
        for i in range(n_dbg):
            t = truth[i]
            p = pred[i]
            pred_txt = (
                f"pred=({p[0]:.0f},{p[1]:.0f})"
                if np.all(np.isfinite(p))
                else "pred=(none)"
            )
            d_txt = ""
            if np.isfinite(du[i]) and np.isfinite(dv[i]):
                d_txt = f"  Δ=({du[i]:+.0f},{dv[i]:+.0f})px"
            e_txt = f"  |err|={err_mm[i]:.1f}mm" if np.isfinite(err_mm[i]) else ""
            print(
                f"  [{i + 1:02d}] truth=({t[0]:.0f},{t[1]:.0f})  "
                f"{pred_txt}{d_txt}{e_txt}"
            )
        ok = np.isfinite(du) & np.isfinite(dv) & np.isfinite(err_mm)
        if int(np.count_nonzero(ok)) >= 2:
            print(
                f"  mean Δu={float(du[ok].mean()):+.0f}px  "
                f"Δv={float(dv[ok].mean()):+.0f}px  "
                f"|err|={float(err_mm[ok].mean()):.1f}mm"
            )
            # Vertical compression signature: top clicks Δv>0, bottom Δv<0.
            top = ok & (truth[:, 1] < float(np.nanmedian(truth[:, 1])))
            bot = ok & (truth[:, 1] >= float(np.nanmedian(truth[:, 1])))
            if int(np.count_nonzero(top)) and int(np.count_nonzero(bot)):
                print(
                    f"  top-half mean Δv={float(dv[top].mean()):+.0f}px  "
                    f"bottom-half mean Δv={float(dv[bot].mean()):+.0f}px"
                )

    if pose_suspect:
        print(
            "FAIL: fix ArUco pose stability first (hold head, 4 markers, re-dump) "
            "before chasing FOV/extrinsics."
        )
        return 3
    if result.mean_residual_mm > 50:
        print(
            "FAIL: residual still high after re-solve -> bad inputs "
            "(directions/angular scale), not just optimizer noise."
        )
        return 2
    print("OK: residual in a usable range for heatmap MVP (ray ∩ plane).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
