"""Viewpoint / camera-rig pose sampling."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .config import CameraConfig, ViewpointConfig
from .geometry import euler_T, inv_T, look_at_cv, make_T, small_rotation, transform_points

GOLDEN = (math.sqrt(5) - 1) / 2


def rig_T(c: CameraConfig) -> np.ndarray:
    return euler_T(c.rig_rotation_deg, c.rig_translation)


def half_fov(c: CameraConfig) -> float:
    return min(math.atan((c.width / 2) / c.fx), math.atan((c.height / 2) / c.fy))


def _distance(v: ViewpointConfig, cam: CameraConfig, scene_radius: float, rng) -> float:
    if v.distance_mode == "fill_fraction":
        fill = max(v.fill_fraction.sample(rng), 1e-3)
        ang = half_fov(cam) * fill
        return max(scene_radius / math.sin(max(ang, 1e-3)), scene_radius * 1.05 + 0.02)
    return v.radius.sample(rng)


def _angles(v: ViewpointConfig, i: int, n: int, rng, phase: float, scene_el: float):
    """(azimuth, elevation) in degrees for view i of n."""
    a0, a1 = v.azimuth_deg.min, v.azimuth_deg.max
    e0, e1 = sorted((v.elevation_deg.min, v.elevation_deg.max))     # min > max is tolerated like in Range.sample
    if v.mode == "sphere_random":
        s = rng.uniform(math.sin(math.radians(e0)), math.sin(math.radians(e1)))  # uniform on sphere cap
        return v.azimuth_deg.sample(rng), math.degrees(math.asin(np.clip(s, -1, 1)))
    if v.mode == "sphere_fibonacci":
        s0, s1 = math.sin(math.radians(e0)), math.sin(math.radians(e1))
        s = s0 + (i + 0.5) / n * (s1 - s0)
        az = a0 + ((i * GOLDEN + phase) % 1.0) * (a1 - a0)
        return az, math.degrees(math.asin(np.clip(s, -1, 1)))
    if v.mode == "grid":
        na, ne = max(1, v.grid_azimuth_steps), max(1, v.grid_elevation_steps)
        ia, ie = i % na, (i // na) % ne
        full = abs((a1 - a0) - 360) < 1e-6
        az = a0 + (ia / na if full else (ia / max(1, na - 1))) * (a1 - a0)
        el = e0 + (ie / max(1, ne - 1)) * (e1 - e0) if ne > 1 else 0.5 * (e0 + e1)
        return az, el
    if v.mode == "orbit":
        return a0 + phase * 360 + (i / max(1, n)) * (a1 - a0), scene_el
    return 0.0, 45.0


def views_per_scene(v: ViewpointConfig) -> int:
    if v.mode == "grid":
        return max(1, v.grid_azimuth_steps) * max(1, v.grid_elevation_steps)
    return max(1, v.views_per_scene)


def _visible_count(T_world_cam: np.ndarray, cam: CameraConfig, centers: np.ndarray) -> int:
    if len(centers) == 0:
        return 0
    Pc = transform_points(inv_T(T_world_cam), centers)
    ok = Pc[:, 2] > 0.01
    uv = Pc[:, :2] / np.clip(Pc[:, 2:3], 1e-6, None) * [cam.fx, cam.fy] + [cam.cx, cam.cy]
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    return int(ok.sum())


def sample_scene_views(v: ViewpointConfig, cams: List[CameraConfig], center: np.ndarray, radius: float,
                       obj_centers: np.ndarray, rng: np.random.Generator) -> List[List[np.ndarray]]:
    """Returns views[view_idx][cam_idx] = T_world_cam (OpenCV)."""
    n = views_per_scene(v)
    phase = rng.uniform()
    scene_el = v.elevation_deg.sample(rng)
    orbit: Optional[dict] = {} if v.mode == "orbit" else None    # one target + distance per scene (a true orbit)
    out: List[List[np.ndarray]] = []
    for i in range(n):
        best, best_score = None, -1
        for attempt in range(max(1, v.max_view_attempts)):
            Ts = _sample_one(v, cams, center, radius, obj_centers, rng, i, n, phase, scene_el, orbit,
                             retry=attempt > 0)
            score = min(_visible_count(T, c, obj_centers) for T, c in zip(Ts, cams))
            if score > best_score:
                best, best_score = Ts, score
            if score >= v.min_visible_objects or v.mode in ("grid", "fixed"):
                break
        out.append(best)
    return out


def _target(v: ViewpointConfig, center, obj_centers, rng) -> np.ndarray:
    if v.target == "origin":
        t = np.zeros(3)
    elif v.target == "random_object" and len(obj_centers):
        t = obj_centers[int(rng.integers(len(obj_centers)))].copy()
    else:
        t = np.asarray(center, float).copy()
    if v.look_at_jitter > 0:
        t = t + rng.normal(0, v.look_at_jitter, 3)
    return t


def _fixed_poses(v, cams, center, rng) -> List[np.ndarray]:
    """Calibrated world poses (+ jitter). Cameras without a world pose follow the rig: their pose comes from the
    first calibrated camera (or a default pose) and the rig extrinsics. A rigid rig is jittered as one body."""
    ref = next((c for c in cams if c.world_pose is not None), None)
    T_ref = np.asarray(ref.world_pose, float) if ref is not None else look_at_cv([0.0, -0.5, 0.5], center)
    T_world_rig = T_ref @ inv_T(rig_T(ref if ref is not None else cams[0]))
    Ts = [np.asarray(c.world_pose, float) if c.world_pose is not None else T_world_rig @ rig_T(c) for c in cams]
    if v.fixed_jitter_translation <= 0 and v.fixed_jitter_rotation_deg <= 0:
        return Ts

    def jitter():
        return make_T(small_rotation(rng, v.fixed_jitter_rotation_deg), rng.normal(0, v.fixed_jitter_translation, 3))
    if v.rig_mode == "rigid" or len(cams) == 1:
        M = Ts[0] @ jitter() @ inv_T(Ts[0])     # one motion of the whole rig: relative camera poses stay exact
        return [M @ T for T in Ts]
    return [T @ jitter() for T in Ts]


def _sample_one(v, cams, center, radius, obj_centers, rng, i, n, phase, scene_el, orbit, retry) -> List[np.ndarray]:
    if v.mode == "fixed":
        return _fixed_poses(v, cams, center, rng)

    def one(cam_for_dist: CameraConfig) -> np.ndarray:
        if v.mode == "sphere_random" or retry:
            az, el = _angles(v.model_copy(update={"mode": "sphere_random"}) if retry and v.mode != "sphere_random" else v,
                             i, n, rng, phase, scene_el)
        else:
            az, el = _angles(v, i, n, rng, phase, scene_el)
        if orbit:                               # later views of an orbit: same centre, same distance
            d, tgt = orbit["d"], orbit["tgt"]
        else:
            d = _distance(v, cam_for_dist, radius, rng)
            tgt = _target(v, center, obj_centers, rng)
            if orbit is not None:
                orbit.update(d=d, tgt=tgt)
        a, e = math.radians(az), math.radians(el)
        eye = tgt + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        return look_at_cv(eye, tgt, roll_deg=v.roll_deg.sample(rng))

    if v.rig_mode == "rigid" or len(cams) == 1:
        T_world_rig = one(cams[0])
        return [T_world_rig @ rig_T(c) for c in cams]
    return [one(c) for c in cams]
