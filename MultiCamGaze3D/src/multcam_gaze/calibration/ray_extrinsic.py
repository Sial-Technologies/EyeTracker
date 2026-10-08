"""Solve T_front←eye so gaze rays pass near known screen points."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.core.transform import Transform


@dataclass(frozen=True)
class RayExtrinsicSample:
    """One look-at sample: eye-tracker ray + known target in front frame."""

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]
    target_front: NDArray[np.float64]
    target_uv: tuple[float, float] | None = None
    # Per-frame / averaged ArUco quality at capture (None on older dumps).
    aruco_reproj_px: float | None = None
    aruco_markers: int | None = None


@dataclass(frozen=True)
class RayExtrinsicResult:
    front_from_eye: Transform
    mean_residual_mm: float
    mean_residual_deg: float
    residuals_mm: NDArray[np.float64]
    direction_scale: float = 1.0
    scale_yaw: float = 1.0
    scale_pitch: float = 1.0
    screen_affine_2x3: NDArray[np.float64] | None = None
    mean_residual_px: float | None = None
    n_samples_used: int = 0
    # Diagnostic-only 2D maps (not used for pure-3D heatmap).
    yaw_pitch_to_uv_2x3: NDArray[np.float64] | None = None
    yaw_pitch_uv_rmse_px: float | None = None


def apply_yaw_pitch_uv(
    direction_front: NDArray[np.float64],
    yaw_pitch_to_uv_2x3: NDArray[np.float64] | None,
) -> tuple[float, float] | None:
    """Diagnostic: map front-frame direction → UV via yaw/pitch affine (not heatmap)."""
    if yaw_pitch_to_uv_2x3 is None:
        return None
    yaw, pitch = direction_to_yaw_pitch(direction_front)
    a = np.asarray(yaw_pitch_to_uv_2x3, dtype=np.float64).reshape(2, 3)
    out = a @ np.array([yaw, pitch, 1.0], dtype=np.float64)
    return float(out[0]), float(out[1])


def fit_yaw_pitch_to_uv(
    samples: list[RayExtrinsicSample],
    rotation: NDArray[np.float64],
    *,
    scale_yaw: float = 1.0,
    scale_pitch: float = 1.0,
) -> tuple[NDArray[np.float64] | None, float | None]:
    """Diagnostic least-squares [yaw,pitch,1] → [u,v] (not used for pure-3D heatmap)."""
    usable = [s for s in samples if s.target_uv is not None]
    if len(usable) < 3:
        return None, None
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    rows: list[list[float]] = []
    uvs: list[list[float]] = []
    for s in usable:
        d = scale_gaze_direction(
            s.direction, 1.0, scale_yaw=scale_yaw, scale_pitch=scale_pitch
        )
        d_f = r @ d
        yaw, pitch = direction_to_yaw_pitch(d_f)
        rows.append([yaw, pitch, 1.0])
        assert s.target_uv is not None
        uvs.append([s.target_uv[0], s.target_uv[1]])
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(uvs, dtype=np.float64)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    affine = beta.T  # 2x3
    err = x @ beta - y
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return affine.astype(np.float64), rmse


def plane_distances_mm(samples: list[RayExtrinsicSample]) -> NDArray[np.float64]:
    """Signed distance of each ``target_front`` from the consensus screen plane."""
    if not samples:
        return np.zeros(0, dtype=np.float64)
    pts = np.stack(
        [np.asarray(s.target_front, dtype=np.float64).reshape(3) for s in samples]
    )
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=np.float64)
    origin, _, _, normal = _fit_plane(pts)
    return np.asarray(
        [float(np.dot(normal, p - origin)) for p in pts], dtype=np.float64
    )


def p_front_z_stats(
    samples: list[RayExtrinsicSample],
) -> tuple[float, float, float]:
    """Return (mean, std, range) of ``target_front`` Z across samples."""
    if not samples:
        return float("nan"), float("nan"), float("nan")
    zs = np.array(
        [float(np.asarray(s.target_front, dtype=np.float64).reshape(3)[2]) for s in samples],
        dtype=np.float64,
    )
    return float(zs.mean()), float(zs.std()), float(zs.max() - zs.min())


def _normalize(v: NDArray[np.float64]) -> NDArray[np.float64] | None:
    d = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None
    return d / n


def direction_to_yaw_pitch(direction: NDArray[np.float64]) -> tuple[float, float]:
    """OpenCV camera frame: +Z forward, +Y down, +X right. Yaw right+, pitch down+."""
    d = _normalize(direction)
    if d is None:
        return 0.0, 0.0
    yaw = float(np.arctan2(d[0], d[2]))
    pitch = float(np.arctan2(d[1], float(np.hypot(d[0], d[2]))))
    return yaw, pitch


def yaw_pitch_to_direction(yaw: float, pitch: float) -> NDArray[np.float64]:
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    return np.array([sy * cp, sp, cy * cp], dtype=np.float64)


def scale_gaze_direction(
    direction: NDArray[np.float64],
    scale: float,
    *,
    scale_yaw: float | None = None,
    scale_pitch: float | None = None,
) -> NDArray[np.float64]:
    """Scale gaze angles (FOV residual). Prefer yaw/pitch scales; ``scale`` is fallback."""
    sy = float(scale if scale_yaw is None else scale_yaw)
    sp = float(scale if scale_pitch is None else scale_pitch)
    yaw, pitch = direction_to_yaw_pitch(direction)
    out = yaw_pitch_to_direction(yaw * sy, pitch * sp)
    n = _normalize(out)
    return out if n is None else n


def rotation_aligning_a_to_b(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """R such that R @ a ≈ b (Rodrigues / 180° fallback)."""
    aa = _normalize(a)
    bb = _normalize(b)
    if aa is None or bb is None:
        return np.eye(3, dtype=np.float64)
    v = np.cross(aa, bb)
    c = float(np.dot(aa, bb))
    s = float(np.linalg.norm(v))
    if s < 1e-6:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(aa[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(aa, axis)
        v = v / np.linalg.norm(v)
        return (2.0 * np.outer(v, v) - np.eye(3)).astype(np.float64)
    vx, vy, vz = v / s
    k = np.array(
        [[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]],
        dtype=np.float64,
    )
    return (np.eye(3) + k * s + (k @ k) * ((1.0 - c) / (s * s))).astype(np.float64)


def kabsch_rotation(
    sources: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Best R mapping source unit dirs → target unit dirs (Wahba/Kabsch)."""
    a = np.asarray(sources, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return r


def filter_coplanar_samples(
    samples: list[RayExtrinsicSample],
    *,
    max_plane_dist_mm: float = 45.0,
) -> list[RayExtrinsicSample]:
    """Drop look-ats whose ArUco P_front is far from the consensus screen plane."""
    if len(samples) < 4:
        return list(samples)
    pts = np.stack(
        [np.asarray(s.target_front, dtype=np.float64).reshape(3) for s in samples]
    )
    origin, _, _, normal = _fit_plane(pts)
    kept: list[RayExtrinsicSample] = []
    for s, p in zip(samples, pts):
        dist = abs(float(np.dot(normal, p - origin)))
        if dist <= max_plane_dist_mm:
            kept.append(s)
    if len(kept) < 3:
        return list(samples)
    return kept


def robust_average_samples(
    samples: list[RayExtrinsicSample],
    *,
    trim_frac: float = 0.2,
) -> RayExtrinsicSample:
    """Collapse a burst of looks at one target into one sample.

    Trims the most extreme directions (angular distance from the circular mean),
    then averages remaining origins / directions / targets. Keeps at least 3
    samples when possible so a short burst still yields a usable mean.
    """
    if not samples:
        raise ValueError("Need at least one sample to average")
    if len(samples) == 1:
        s = samples[0]
        return RayExtrinsicSample(
            origin=np.asarray(s.origin, dtype=np.float64).reshape(3).copy(),
            direction=np.asarray(s.direction, dtype=np.float64).reshape(3).copy(),
            target_front=np.asarray(s.target_front, dtype=np.float64).reshape(3).copy(),
            target_uv=s.target_uv,
            aruco_reproj_px=s.aruco_reproj_px,
            aruco_markers=s.aruco_markers,
        )

    dirs = np.stack(
        [np.asarray(s.direction, dtype=np.float64).reshape(3) for s in samples]
    )
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs = dirs / norms
    mean_dir = dirs.mean(axis=0)
    mn = float(np.linalg.norm(mean_dir))
    if mn < 1e-12:
        mean_dir = dirs[0]
    else:
        mean_dir = mean_dir / mn

    # Angular distance from provisional mean; trim the farthest fraction.
    dots = np.clip(dirs @ mean_dir, -1.0, 1.0)
    ang = np.arccos(dots)
    n = len(samples)
    trim = float(np.clip(trim_frac, 0.0, 0.45))
    keep_n = max(3, int(round(n * (1.0 - trim)))) if n >= 3 else n
    keep_n = min(keep_n, n)
    keep_idx = np.argsort(ang)[:keep_n]

    origins = np.stack(
        [np.asarray(samples[i].origin, dtype=np.float64).reshape(3) for i in keep_idx]
    )
    dirs_k = dirs[keep_idx]
    targets = np.stack(
        [
            np.asarray(samples[i].target_front, dtype=np.float64).reshape(3)
            for i in keep_idx
        ]
    )
    d_avg = dirs_k.mean(axis=0)
    dn = float(np.linalg.norm(d_avg))
    if dn < 1e-12:
        d_avg = dirs_k[0]
    else:
        d_avg = d_avg / dn
    uv = next((samples[i].target_uv for i in keep_idx if samples[i].target_uv), None)
    kept = [samples[i] for i in keep_idx]
    reprojs = [s.aruco_reproj_px for s in kept if s.aruco_reproj_px is not None]
    markers = [s.aruco_markers for s in kept if s.aruco_markers is not None]
    return RayExtrinsicSample(
        origin=origins.mean(axis=0),
        direction=d_avg,
        target_front=targets.mean(axis=0),
        target_uv=uv,
        aruco_reproj_px=float(np.mean(reprojs)) if reprojs else None,
        aruco_markers=int(min(markers)) if markers else None,
    )


@dataclass(frozen=True)
class BurstStability:
    ok: bool
    n: int
    rms_angle_deg: float
    max_angle_deg: float
    reason: str
    mean_aruco_reproj_px: float = float("nan")
    min_aruco_markers: int = 0
    p_front_z_range_mm: float = float("nan")


def direction_angular_stats_deg(
    samples: list[RayExtrinsicSample],
) -> tuple[float, float]:
    """RMS and max angular deviation (deg) of directions from their circular mean."""
    if not samples:
        return 180.0, 180.0
    dirs = np.stack(
        [np.asarray(s.direction, dtype=np.float64).reshape(3) for s in samples]
    )
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs = dirs / norms
    mean_dir = dirs.mean(axis=0)
    mn = float(np.linalg.norm(mean_dir))
    if mn < 1e-12:
        return 180.0, 180.0
    mean_dir = mean_dir / mn
    dots = np.clip(dirs @ mean_dir, -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    rms = float(np.sqrt(np.mean(ang * ang)))
    return rms, float(np.max(ang))


def assess_burst_stability(
    samples: list[RayExtrinsicSample],
    *,
    min_samples: int,
    max_rms_deg: float,
    max_angle_deg: float,
    min_aruco_markers: int = 0,
    max_aruco_reproj_px: float | None = None,
    max_pfront_z_range_mm: float | None = None,
) -> BurstStability:
    """Accept a look-at burst only if gaze and ArUco pose are stable enough."""
    n = len(samples)
    _, _, z_range = p_front_z_stats(samples)
    reprojs = [s.aruco_reproj_px for s in samples if s.aruco_reproj_px is not None]
    markers = [s.aruco_markers for s in samples if s.aruco_markers is not None]
    mean_reproj = float(np.mean(reprojs)) if reprojs else float("nan")
    min_markers = int(min(markers)) if markers else 0

    def _fail(reason: str, rms: float = 180.0, amax: float = 180.0) -> BurstStability:
        return BurstStability(
            False,
            n,
            rms,
            amax,
            reason,
            mean_aruco_reproj_px=mean_reproj,
            min_aruco_markers=min_markers,
            p_front_z_range_mm=z_range,
        )

    if n < min_samples:
        return _fail(f"too few valid frames ({n}<{min_samples})")
    rms, amax = direction_angular_stats_deg(samples)
    if rms > max_rms_deg:
        return _fail(f"gaze RMS {rms:.1f}° > {max_rms_deg:.1f}°", rms, amax)
    if amax > max_angle_deg:
        return _fail(f"gaze max {amax:.1f}° > {max_angle_deg:.1f}°", rms, amax)
    if min_aruco_markers > 0 and min_markers < min_aruco_markers:
        return _fail(
            f"ArUco markers {min_markers} < {min_aruco_markers}",
            rms,
            amax,
        )
    if (
        max_aruco_reproj_px is not None
        and reprojs
        and mean_reproj > max_aruco_reproj_px
    ):
        return _fail(
            f"ArUco reproj {mean_reproj:.2f}px > {max_aruco_reproj_px:.2f}px",
            rms,
            amax,
        )
    if (
        max_pfront_z_range_mm is not None
        and np.isfinite(z_range)
        and z_range > max_pfront_z_range_mm
    ):
        return _fail(
            f"P_front Z range {z_range:.1f}mm > {max_pfront_z_range_mm:.1f}mm",
            rms,
            amax,
        )
    return BurstStability(
        True,
        n,
        rms,
        amax,
        "ok",
        mean_aruco_reproj_px=mean_reproj,
        min_aruco_markers=min_markers,
        p_front_z_range_mm=z_range,
    )


def point_to_ray_distance(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    point: NDArray[np.float64],
) -> float:
    """Perpendicular distance from point to ray; large penalty if behind origin."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        return 1e6
    d = d / dn
    to_p = p - o
    lam = float(np.dot(to_p, d))
    if lam < 1e-6:
        return 1e6
    closest = o + lam * d
    return float(np.linalg.norm(p - closest))


def angular_error_deg(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    point: NDArray[np.float64],
) -> float:
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    tn = float(np.linalg.norm(p - o))
    if dn < 1e-12 or tn < 1e-12:
        return 180.0
    d = d / dn
    to_p = (p - o) / tn
    c = float(np.clip(np.dot(d, to_p), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def apply_screen_affine(
    uv: tuple[float, float],
    affine_2x3: NDArray[np.float64] | None,
) -> tuple[float, float]:
    """Map predicted screen pixels through an optional 2x3 affine polish."""
    if affine_2x3 is None:
        return float(uv[0]), float(uv[1])
    a = np.asarray(affine_2x3, dtype=np.float64).reshape(2, 3)
    x = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    out = a @ x
    return float(out[0]), float(out[1])


# Soft prior: yaw/pitch scales absorb residual FOV error only — prefer ~1 after
# sensor-space remap + calibrated IR K. Weight is in the same units as mean_mm.
SCALE_PRIOR_WEIGHT = 40.0


def _eval_r_scales(
    samples: list[RayExtrinsicSample],
    rotation: NDArray[np.float64],
    scale_yaw: float,
    scale_pitch: float,
    *,
    ray_origin_front: NDArray[np.float64] | None = None,
) -> tuple[float, float, float]:
    """Score R + yaw/pitch scales; ray origin defaults to front-camera center."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    eye = (
        np.zeros(3, dtype=np.float64)
        if ray_origin_front is None
        else np.asarray(ray_origin_front, dtype=np.float64).reshape(3)
    )
    dists: list[float] = []
    angs: list[float] = []
    for s in samples:
        d_eye = scale_gaze_direction(
            s.direction, 1.0, scale_yaw=scale_yaw, scale_pitch=scale_pitch
        )
        d = r @ d_eye
        p = np.asarray(s.target_front, dtype=np.float64).reshape(3)
        dists.append(point_to_ray_distance(eye, d, p))
        angs.append(angular_error_deg(eye, d, p))
    mean_mm = float(np.mean(dists))
    mean_deg = float(np.mean(angs))
    prior = SCALE_PRIOR_WEIGHT * (
        (float(scale_yaw) - 1.0) ** 2 + (float(scale_pitch) - 1.0) ** 2
    )
    cost = mean_mm + 25.0 * mean_deg + prior
    return mean_mm, mean_deg, cost


def _fit_plane(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (origin, e1, e2, normal) for a best-fit plane through points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    origin = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - origin, full_matrices=False)
    e1 = vh[0]
    e2 = vh[1]
    normal = vh[2]
    if float(np.linalg.norm(normal)) < 1e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / np.linalg.norm(normal)
    return origin, e1, e2, normal


def _ray_plane_hit(
    origin: NDArray[np.float64],
    direction: NDArray[np.float64],
    plane_origin: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-12:
        return None
    d = d / dn
    denom = float(np.dot(normal, d))
    if abs(denom) < 1e-9:
        return None
    lam = float(np.dot(normal, plane_origin - origin)) / denom
    if lam < 1e-6:
        return None
    return origin + lam * d


def fit_screen_affine_2x3(
    samples: list[RayExtrinsicSample],
    front_from_eye: Transform,
    *,
    direction_scale: float = 1.0,
    scale_yaw: float | None = None,
    scale_pitch: float | None = None,
) -> tuple[NDArray[np.float64] | None, float | None]:
    """Fit UV polish from rigid-ray predictions to known look-at pixels."""
    usable = [s for s in samples if s.target_uv is not None]
    if len(usable) < 3:
        return None, None

    targets = np.stack(
        [np.asarray(s.target_front, dtype=np.float64).reshape(3) for s in usable]
    )
    plane_o, e1, e2, normal = _fit_plane(targets)

    st_rows: list[list[float]] = []
    uv_rows: list[list[float]] = []
    for s in usable:
        p = np.asarray(s.target_front, dtype=np.float64).reshape(3)
        delta = p - plane_o
        st_rows.append([float(np.dot(delta, e1)), float(np.dot(delta, e2)), 1.0])
        assert s.target_uv is not None
        uv_rows.append([s.target_uv[0], s.target_uv[1]])
    st = np.asarray(st_rows, dtype=np.float64)
    uv = np.asarray(uv_rows, dtype=np.float64)
    st_to_uv, _, _, _ = np.linalg.lstsq(st, uv, rcond=None)

    pred: list[list[float]] = []
    true: list[list[float]] = []
    eye = front_from_eye.translation
    for s in usable:
        d = front_from_eye.apply_direction(
            scale_gaze_direction(
                s.direction,
                direction_scale,
                scale_yaw=scale_yaw,
                scale_pitch=scale_pitch,
            )
        )
        hit = _ray_plane_hit(eye, d, plane_o, normal)
        if hit is None:
            continue
        delta = hit - plane_o
        st_h = np.array(
            [float(np.dot(delta, e1)), float(np.dot(delta, e2)), 1.0], dtype=np.float64
        )
        uv_pred = st_h @ st_to_uv
        assert s.target_uv is not None
        pred.append([float(uv_pred[0]), float(uv_pred[1]), 1.0])
        true.append([s.target_uv[0], s.target_uv[1]])
    if len(pred) < 3:
        return None, None

    p_mat = np.asarray(pred, dtype=np.float64)
    t_mat = np.asarray(true, dtype=np.float64)
    affine, _, _, _ = np.linalg.lstsq(p_mat, t_mat, rcond=None)
    affine = affine.T  # 2x3
    err = p_mat @ affine.T - t_mat
    mean_px = float(np.mean(np.linalg.norm(err, axis=1)))
    return affine.astype(np.float64), mean_px


def _gaze_expected_pairs(
    samples: list[RayExtrinsicSample],
    eye_origin: NDArray[np.float64],
    *,
    project_to_consensus_plane: bool = True,
    coplanar_max_dist_mm: float | None = 80.0,
) -> tuple[list[RayExtrinsicSample], NDArray[np.float64], NDArray[np.float64]]:
    """Build (used samples, gaze dirs, expected dirs) for a candidate eye origin.

    When ``project_to_consensus_plane`` is true (default single-pose calib), targets
    are projected onto one fit plane to kill PnP depth jitter. For multi-distance
    dumps, set it false so depth diversity can constrain ``eye_origin``.
    """
    used = list(samples)
    if coplanar_max_dist_mm is not None:
        used = filter_coplanar_samples(used, max_plane_dist_mm=float(coplanar_max_dist_mm))
    if len(used) < 3:
        raise ValueError("Need at least 3 look-at samples after coplanar filter")

    pts = np.stack(
        [np.asarray(s.target_front, dtype=np.float64).reshape(3) for s in used]
    )
    plane_o = plane_n = None
    if project_to_consensus_plane:
        plane_o, _, _, plane_n = _fit_plane(pts)

    gazes: list[NDArray[np.float64]] = []
    expected: list[NDArray[np.float64]] = []
    out_samples: list[RayExtrinsicSample] = []
    eye = np.asarray(eye_origin, dtype=np.float64).reshape(3)
    for s in used:
        g = _normalize(np.asarray(s.direction, dtype=np.float64))
        p = np.asarray(s.target_front, dtype=np.float64).reshape(3)
        if project_to_consensus_plane:
            assert plane_o is not None and plane_n is not None
            p = p - float(np.dot(p - plane_o, plane_n)) * plane_n
        e = _normalize(p - eye)
        if g is None or e is None:
            continue
        gazes.append(g)
        expected.append(e)
        out_samples.append(
            RayExtrinsicSample(
                s.origin,
                s.direction,
                p,
                s.target_uv,
                aruco_reproj_px=s.aruco_reproj_px,
                aruco_markers=s.aruco_markers,
            )
        )
    if len(gazes) < 3:
        raise ValueError("Need at least 3 valid direction/target pairs")
    return out_samples, np.stack(gazes), np.stack(expected)


def kabsch_residual_mm_for_eye_origin(
    samples: list[RayExtrinsicSample],
    ray_origin_front: NDArray[np.float64],
    *,
    project_to_consensus_plane: bool = True,
    coplanar_max_dist_mm: float | None = 80.0,
) -> float:
    """Fast mean point-to-ray miss (mm) for a candidate E: Kabsch R, scales=1."""
    eye = np.asarray(ray_origin_front, dtype=np.float64).reshape(3)
    used, g_mat, e_mat = _gaze_expected_pairs(
        samples,
        eye,
        project_to_consensus_plane=project_to_consensus_plane,
        coplanar_max_dist_mm=coplanar_max_dist_mm,
    )
    r = kabsch_rotation(g_mat, e_mat)
    dists = [
        point_to_ray_distance(eye, r @ g_mat[i], s.target_front)
        for i, s in enumerate(used)
    ]
    return float(np.mean(dists))


@dataclass(frozen=True)
class EyeOriginRefineResult:
    """Offline grid-search for ``eye_center_front_mm`` from look-at dumps."""

    eye_origin_front: NDArray[np.float64]
    seed: NDArray[np.float64]
    residual_mm: float
    seed_residual_mm: float
    n_evals: int
    solve: RayExtrinsicResult | None = None


# mm of Kabsch residual traded for 1 mm of |E - seed| (keeps CAD-plausible E).
EYE_ORIGIN_SEED_PRIOR_PER_MM = 0.12


def _as_sample_groups(
    samples: list[RayExtrinsicSample] | list[list[RayExtrinsicSample]],
) -> list[list[RayExtrinsicSample]]:
    if not samples:
        raise ValueError("Need at least one sample group")
    first = samples[0]
    if isinstance(first, RayExtrinsicSample):
        return [list(samples)]  # type: ignore[arg-type]
    return [list(g) for g in samples]  # type: ignore[arg-type]


def refine_eye_origin_front(
    samples: list[RayExtrinsicSample] | list[list[RayExtrinsicSample]],
    *,
    seed: NDArray[np.float64],
    half_extent_mm: float = 40.0,
    step_mm: float = 5.0,
    fine_half_extent_mm: float = 10.0,
    fine_step_mm: float = 2.0,
    project_to_consensus_plane: bool = True,
    coplanar_max_dist_mm: float | None = 80.0,
    run_full_solve: bool = True,
    seed_prior_per_mm: float = EYE_ORIGIN_SEED_PRIOR_PER_MM,
) -> EyeOriginRefineResult:
    """Coarse→fine search for eyeball origin E that minimizes Kabsch ray miss.

    Pass a list of sample groups (e.g. near dump, far dump) to score the mean
    miss across poses — each group may use consensus-plane projection. A flat
    sample list is treated as one group. ``seed_prior_per_mm`` penalizes
    drifting far from the CAD/tape estimate; fine search stays inside the
    coarse box around ``seed``.
    """
    groups = _as_sample_groups(samples)
    flat = [s for g in groups for s in g]
    if len(flat) < 3:
        raise ValueError("Need at least 3 look-at samples to refine eye origin")
    for i, g in enumerate(groups):
        if len(g) < 3:
            raise ValueError(f"Sample group {i} needs at least 3 look-ats (have {len(g)})")

    seed_e = np.asarray(seed, dtype=np.float64).reshape(3).copy()
    prior_w = float(max(seed_prior_per_mm, 0.0))
    max_radius = float(max(half_extent_mm, 0.0)) + 1e-6

    def _miss(e: NDArray[np.float64]) -> float:
        vals = [
            kabsch_residual_mm_for_eye_origin(
                g,
                e,
                project_to_consensus_plane=project_to_consensus_plane,
                coplanar_max_dist_mm=coplanar_max_dist_mm,
            )
            for g in groups
        ]
        return float(np.mean(vals))

    def _cost(e: NDArray[np.float64]) -> float:
        miss = _miss(e)
        if prior_w <= 0.0:
            return miss
        return miss + prior_w * float(np.linalg.norm(e - seed_e))

    def _grid_search(
        center: NDArray[np.float64],
        half: float,
        step: float,
        *,
        clamp_to_seed_box: bool,
    ) -> tuple[NDArray[np.float64], float, int]:
        half = float(max(half, 0.0))
        step = float(max(step, 0.5))
        axes = [np.arange(-half, half + 0.5 * step, step) for _ in range(3)]
        best_e = center.copy()
        best_c = _cost(center)
        n = 1
        for dx in axes[0]:
            for dy in axes[1]:
                for dz in axes[2]:
                    if dx == 0.0 and dy == 0.0 and dz == 0.0:
                        continue
                    e = center + np.array([dx, dy, dz], dtype=np.float64)
                    if clamp_to_seed_box and float(np.linalg.norm(e - seed_e)) > max_radius:
                        continue
                    c = _cost(e)
                    n += 1
                    if c < best_c:
                        best_c = c
                        best_e = e
        return best_e, best_c, n

    seed_miss = _miss(seed_e)
    best_e, _, n0 = _grid_search(
        seed_e, half_extent_mm, step_mm, clamp_to_seed_box=False
    )
    best_e, _, n1 = _grid_search(
        best_e, fine_half_extent_mm, fine_step_mm, clamp_to_seed_box=True
    )
    n_evals = n0 + n1
    best_miss = _miss(best_e)

    full: RayExtrinsicResult | None = None
    if run_full_solve:
        # Joint full solve without plane squash when multiple depth groups.
        joint_plane = project_to_consensus_plane and len(groups) == 1
        joint_coplanar = coplanar_max_dist_mm if joint_plane else None
        full = solve_front_from_eye(
            flat,
            ray_origin_front=best_e,
            project_to_consensus_plane=joint_plane,
            coplanar_max_dist_mm=joint_coplanar,
        )

    return EyeOriginRefineResult(
        eye_origin_front=best_e,
        seed=seed_e,
        residual_mm=float(best_miss),
        seed_residual_mm=float(seed_miss),
        n_evals=int(n_evals),
        solve=full,
    )


def solve_front_from_eye(
    samples: list[RayExtrinsicSample],
    *,
    eye_frame: str = "left_eye",
    front_frame: str = "front",
    ray_origin_front: NDArray[np.float64] | None = None,
    project_to_consensus_plane: bool = True,
    coplanar_max_dist_mm: float | None = 80.0,
) -> RayExtrinsicResult:
    """Solve R (+ yaw/pitch scales) so gaze rays pass near look-at points.

    ``ray_origin_front`` is the eyeball center in the front frame (mm). Default
    is the front-camera origin. Prefer a mount/CAD estimate
    (``eye_center_front_mm`` in camera_setup.json). Orlosky's virtual sphere
    center is never used as metric origin. Directions come from the IR tracker;
    translation of the returned transform is ``ray_origin_front``.

    Set ``project_to_consensus_plane=False`` (and usually
    ``coplanar_max_dist_mm=None``) when combining multi-distance look-at dumps
    so depth diversity is preserved for eye-origin search.
    """
    eye_origin = (
        np.zeros(3, dtype=np.float64)
        if ray_origin_front is None
        else np.asarray(ray_origin_front, dtype=np.float64).reshape(3).copy()
    )
    if len(samples) < 3:
        raise ValueError("Need at least 3 look-at samples to solve front_from_eye")

    used, g_mat, e_mat = _gaze_expected_pairs(
        samples,
        eye_origin,
        project_to_consensus_plane=project_to_consensus_plane,
        coplanar_max_dist_mm=coplanar_max_dist_mm,
    )
    r0 = kabsch_rotation(g_mat, e_mat)

    # Also try aligning mean gaze to mean expected (GazeScreen3D center-style seed).
    r_mean = rotation_aligning_a_to_b(g_mat.mean(axis=0), e_mat.mean(axis=0))
    # And the classic IR 180° about X.
    r_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

    best = {
        "R": r0.copy(),
        "sy": 1.0,
        "sp": 1.0,
        "cost": 1e18,
        "mean_mm": 1e18,
        "mean_deg": 180.0,
    }
    scale_grid = (
        0.12,
        0.15,
        0.2,
        0.25,
        0.35,
        0.5,
        0.65,
        0.8,
        0.9,
        1.0,
        1.15,
        1.35,
        1.6,
        2.0,
        2.4,
    )
    for r_init in (r0, r_mean, r_flip, r_flip @ r0, r0 @ r_flip):
        for sy in scale_grid:
            for sp in scale_grid:
                mean_mm, mean_deg, cost = _eval_r_scales(
                    used, r_init, sy, sp, ray_origin_front=eye_origin
                )
                if cost < best["cost"]:
                    best.update(
                        {
                            "R": r_init.copy(),
                            "sy": float(sy),
                            "sp": float(sp),
                            "cost": cost,
                            "mean_mm": mean_mm,
                            "mean_deg": mean_deg,
                        }
                    )

    r = best["R"].copy()
    sy = float(best["sy"])
    sp = float(best["sp"])
    rvec, _ = cv2.Rodrigues(r)

    for step_r, step_s in ((0.08, 0.08), (0.04, 0.04), (0.02, 0.02), (0.01, 0.01)):
        for _ in range(20):
            improved = False
            for axis in range(3):
                for sign in (-1.0, 0.0, 1.0):
                    dr = np.zeros(3)
                    dr[axis] = sign * step_r
                    r_try, _ = cv2.Rodrigues(rvec.reshape(3) + dr)
                    for dsy in (-step_s, 0.0, step_s):
                        for dsp in (-step_s, 0.0, step_s):
                            sy_t = float(np.clip(sy + dsy, 0.1, 3.0))
                            sp_t = float(np.clip(sp + dsp, 0.1, 3.0))
                            mean_mm, mean_deg, cost = _eval_r_scales(
                                used,
                                r_try,
                                sy_t,
                                sp_t,
                                ray_origin_front=eye_origin,
                            )
                            if cost < best["cost"]:
                                best.update(
                                    {
                                        "R": np.asarray(r_try, dtype=np.float64).copy(),
                                        "sy": sy_t,
                                        "sp": sp_t,
                                        "cost": cost,
                                        "mean_mm": mean_mm,
                                        "mean_deg": mean_deg,
                                    }
                                )
                                r = best["R"].copy()
                                rvec, _ = cv2.Rodrigues(r)
                                sy, sp = best["sy"], best["sp"]
                                improved = True
            if not improved:
                break

    # Translation = eyeball in front/world mm; direction map is R only.
    transform = Transform.from_rotation_translation(
        best["R"], eye_origin, front_frame, eye_frame
    )
    sy, sp = float(best["sy"]), float(best["sp"])
    scale_geom = float(np.sqrt(max(sy * sp, 1e-12)))

    def _mapped_dir(s: RayExtrinsicSample) -> NDArray[np.float64]:
        return transform.apply_direction(
            scale_gaze_direction(s.direction, scale_geom, scale_yaw=sy, scale_pitch=sp)
        )

    residuals = np.array(
        [
            point_to_ray_distance(eye_origin, _mapped_dir(s), s.target_front)
            for s in used
        ],
        dtype=np.float64,
    )
    mean_deg = float(
        np.mean(
            [
                angular_error_deg(eye_origin, _mapped_dir(s), s.target_front)
                for s in used
            ]
        )
    )
    affine, mean_px = fit_screen_affine_2x3(
        used,
        transform,
        direction_scale=scale_geom,
        scale_yaw=sy,
        scale_pitch=sp,
    )
    yp_affine, yp_rmse = fit_yaw_pitch_to_uv(
        used, best["R"], scale_yaw=sy, scale_pitch=sp
    )
    return RayExtrinsicResult(
        front_from_eye=transform,
        mean_residual_mm=float(np.mean(residuals)),
        mean_residual_deg=mean_deg,
        residuals_mm=residuals,
        direction_scale=scale_geom,
        scale_yaw=sy,
        scale_pitch=sp,
        screen_affine_2x3=affine,
        mean_residual_px=mean_px,
        n_samples_used=len(used),
        yaw_pitch_to_uv_2x3=yp_affine,
        yaw_pitch_uv_rmse_px=yp_rmse,
    )
