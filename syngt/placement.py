"""Object placement: free-floating 6D, separated stable poses, physics drops / piles (leaning), bins.

Physics uses MuJoCo with convex hull (or CoACD convex decomposition) collision geometry.
The body frame equals the CAD frame, so simulated poses are exact GT poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .assets import MeshAsset
from .config import PlacementConfig
from .geometry import make_T, random_rotation, rot_z, small_rotation


SEQUENTIAL_DROP_INTERVAL = 0.2   # seconds between parts in "drop sequentially"


@dataclass
class Item:
    asset: MeshAsset
    scale: float = 1.0
    friction: float = 0.6
    density: float = 1000.0


@dataclass
class PlacementResult:
    poses: List[Optional[np.ndarray]]   # T_world_obj per item (None = removed)
    bin_box: Optional[Tuple[float, float, float, float]] = None  # (sx, sy, h, wall) if a bin was used
    sim_steps: int = 0
    settled: bool = True
    note: str = ""                      # set when the requested placement could not be done as configured


def _hull(item: Item) -> np.ndarray:
    return np.asarray(item.asset.hull_vertices, dtype=np.float64) * item.scale


def _hull_center(item: Item) -> np.ndarray:
    """Centre of the part (object frame). The CAD origin may be far away from the part (assembly coordinates)."""
    h = _hull(item)
    return 0.5 * (h.min(0) + h.max(0))


def _bound_radius(item: Item) -> float:
    """Radius of the bounding sphere around the part's centre (not around the CAD origin)."""
    return float(np.linalg.norm(_hull(item) - _hull_center(item), axis=1).max())


def _sample_orientation(item: Item, mode: str, rng: np.random.Generator, yaw_rng=None, tilt=0.0) -> np.ndarray:
    if mode in ("uniform_so3", "random"):
        return random_rotation(rng)
    yaw = yaw_rng.sample(rng) if yaw_rng is not None else rng.uniform(-180, 180)
    if mode in ("stable_pose",):
        poses = item.asset.get_stable_poses() or [(np.eye(3), 1.0)]
        probs = np.array([p for _, p in poses], dtype=np.float64)
        i = int(rng.choice(len(poses), p=probs / probs.sum()))
        R = poses[i][0]
        return rot_z(yaw) @ small_rotation(rng, tilt) @ R
    # upright
    return rot_z(yaw) @ small_rotation(rng, tilt)


# --------------------------------------------------------------------------- #
# MuJoCo
# --------------------------------------------------------------------------- #
def _fmt(a: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a).reshape(-1))


def simulate(items: List[Item], init_poses: List[np.ndarray], cfg: PlacementConfig, rng: np.random.Generator,
             bin_box=None, release_times: Optional[List[float]] = None, max_time: Optional[float] = None
             ) -> Tuple[List[np.ndarray], int, bool]:
    import mujoco

    friction = cfg.friction.sample(rng)
    ts = float(cfg.timestep)
    tc = max(2 * ts, 0.004)
    asset_xml, body_xml = [], []
    for i, (it, T) in enumerate(zip(items, init_poses)):
        geoms = []
        for k, hv in enumerate(it.asset.collision_hulls or [_hull(it) / max(it.scale, 1e-9)]):
            v = np.asarray(hv, dtype=np.float64) * it.scale
            asset_xml.append(f'<mesh name="m{i}_{k}" vertex="{_fmt(v)}" maxhullvert="64"/>')
            mu = 0.5 * (friction + it.friction)
            # priority 1: against the table / bin (priority 0) the part's value is used, not the larger of both
            geoms.append(f'<geom type="mesh" mesh="m{i}_{k}" density="{it.density:.6g}" priority="1" '
                         f'friction="{mu:.4g} 0.005 0.0005"/>')
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
        body_xml.append(f'<body name="b{i}" pos="{_fmt(T[:3, 3])}" quat="{q[3]:.8g} {q[0]:.8g} {q[1]:.8g} {q[2]:.8g}">'
                        f'<freejoint/>{"".join(geoms)}</body>')
    walls = ""
    if bin_box is not None:
        sx, sy, h, w = bin_box
        hx, hy = sx / 2 + w / 2, sy / 2 + w / 2
        for nm, pos, size in (("wx0", (-hx, 0, h / 2), (w / 2, sy / 2 + w, h / 2)),
                              ("wx1", (hx, 0, h / 2), (w / 2, sy / 2 + w, h / 2)),
                              ("wy0", (0, -hy, h / 2), (sx / 2 + w, w / 2, h / 2)),
                              ("wy1", (0, hy, h / 2), (sx / 2 + w, w / 2, h / 2))):
            walls += f'<geom name="{nm}" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}" friction="{friction:.4g} 0.005 0.0005"/>'
    xml = f"""
<mujoco>
  <option timestep="{ts}" integrator="implicitfast" cone="elliptic" impratio="5" gravity="0 0 -9.81"/>
  <default><geom condim="6" solref="{tc:.5g} {cfg.restitution_damping:.3g}" margin="0"/></default>
  <asset>{''.join(asset_xml)}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" friction="{friction:.4g} 0.005 0.0005"/>
    {walls}
    {''.join(body_xml)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = len(items)
    max_time = max_time if max_time is not None else cfg.sim_time

    # sequential release: park bodies far away, teleport at release time
    parked = []
    if release_times is not None:
        for i in range(n):
            if release_times[i] > 0:
                adr = model.jnt_qposadr[model.body_jntadr[i + 1]]
                parked.append(i)
                # far away and high up: until its release the part is in free fall and never touches anything
                # (parked on the floor it would cost contact computation in every step)
                data.qpos[adr:adr + 3] = [100.0 + 3.0 * i, 100.0, 5000.0]
    mujoco.mj_forward(model, data)

    steps, settled = 0, False
    prev_pos = prev_quat = None   # pose at the previous check: resting parts may keep a tiny contact jitter in
    still = 0                     # their velocity (thin tiles) but their pose no longer changes
    max_steps = int(max_time / ts)
    last_release = max(release_times) if release_times else 0.0
    check_every = max(1, int(0.05 / ts))
    while steps < max_steps:
        t = steps * ts
        if parked:
            for i in list(parked):
                if t >= release_times[i]:
                    adr = model.jnt_qposadr[model.body_jntadr[i + 1]]
                    vadr = model.jnt_dofadr[model.body_jntadr[i + 1]]
                    T = init_poses[i]
                    q = Rotation.from_matrix(T[:3, :3]).as_quat()
                    data.qpos[adr:adr + 7] = [*T[:3, 3], q[3], q[0], q[1], q[2]]
                    data.qvel[vadr:vadr + 6] = 0
                    parked.remove(i)
        mujoco.mj_step(model, data)
        steps += 1
        if steps % check_every == 0:
            if not np.all(np.isfinite(data.qpos)):
                raise RuntimeError("physics diverged")
            if t > last_release + 0.2:
                if np.abs(data.qvel).max() < 0.02:
                    settled = True
                    break
                pos, quat = data.xpos[1:].copy(), data.xquat[1:].copy()
                if prev_pos is not None:
                    dpos = float(np.abs(pos - prev_pos).max())
                    dang = float((2 * np.arccos(np.clip(np.abs((quat * prev_quat).sum(1)), 0, 1))).max())
                    still = still + 1 if dpos < 3e-4 and dang < 0.01 else 0
                    if still >= 2:   # 0.1 s without measurable motion
                        settled = True
                        break
                prev_pos, prev_quat = pos, quat
    out = []
    for i in range(n):
        T = np.eye(4)
        T[:3, :3] = data.xmat[i + 1].reshape(3, 3)
        T[:3, 3] = data.xpos[i + 1]
        out.append(T)
    return out, steps, settled


# --------------------------------------------------------------------------- #
def place(items: List[Item], cfg: PlacementConfig, rng: np.random.Generator, floor: bool = True) -> PlacementResult:
    """floor: a table exists at z = 0 (free-floating parts are then kept above it)."""
    mode = cfg.mode
    if not items:
        return PlacementResult([])
    if mode == "free_floating":
        return _place_floating(items, cfg, rng, floor)
    if mode == "stable_separated":
        return _place_separated(items, cfg, rng)
    return _place_physics(items, cfg, rng, use_bin=(mode == "physics_bin"))


def _place_floating(items, cfg, rng, floor: bool = False) -> PlacementResult:
    vol = np.asarray(cfg.float_volume, float)
    c = np.asarray(cfg.float_center, float)
    placed: List[Tuple[np.ndarray, float]] = []
    poses = []
    for it in items:
        r = _bound_radius(it)
        R = _sample_orientation(it, cfg.float_rotation, rng, cfg.yaw, cfg.tilt_jitter_deg)
        pos = None
        for _ in range(200):
            p = c + (rng.uniform(-0.5, 0.5, 3) * vol)
            if floor:   # never cut through the table: the mask would be clipped while the pose says "complete"
                p[2] = max(p[2], r + 1e-3)
            if not cfg.float_avoid_collisions or all(np.linalg.norm(p - q) > (r + rq) * 0.9 for q, rq in placed):
                pos = p
                break
        if pos is None:
            poses.append(None)
            continue
        # position refers to the object's hull centre, not its CAD origin
        hc = _hull_center(it)
        placed.append((pos, r))
        poses.append(make_T(R, pos - R @ hc))
    return PlacementResult(poses)


def _resting_T(it: Item, R: np.ndarray, xy) -> np.ndarray:
    hv = _hull(it) @ R.T
    hc = 0.5 * (hv.min(0) + hv.max(0))
    t = np.array([xy[0] - hc[0], xy[1] - hc[1], -hv[:, 2].min() + 1e-4])
    return make_T(R, t)


def _place_separated(items, cfg, rng) -> PlacementResult:
    from shapely.geometry import MultiPoint, box
    ax, ay = cfg.area_size
    area = box(-ax / 2, -ay / 2, ax / 2, ay / 2)
    placed_polys = []
    poses: List[Optional[np.ndarray]] = []
    for it in items:
        ok = None
        for attempt in range(80):
            R = _sample_orientation(it, "stable_pose", rng, cfg.yaw, cfg.tilt_jitter_deg)
            hv = _hull(it) @ R.T
            fp = MultiPoint(hv[:, :2]).convex_hull
            c = np.asarray(fp.centroid.coords[0])
            minx, miny, maxx, maxy = fp.bounds
            hw, hh = (maxx - minx) / 2, (maxy - miny) / 2
            if hw * 2 > ax or hh * 2 > ay:
                xy = (0.0, 0.0)
            else:
                xy = (rng.uniform(-ax / 2 + hw, ax / 2 - hw), rng.uniform(-ay / 2 + hh, ay / 2 - hh))
            from shapely import affinity
            poly = affinity.translate(fp, xy[0] - (minx + maxx) / 2, xy[1] - (miny + maxy) / 2)
            buf = poly.buffer(cfg.min_spacing / 2)
            if all(not buf.intersects(p) for p in placed_polys):
                ok = (R, xy, buf)
                break
        if ok is None:
            poses.append(None)
            continue
        R, xy, buf = ok
        placed_polys.append(buf)
        poses.append(_resting_T(it, R, xy))
    res = PlacementResult(poses)
    if cfg.settle_with_physics:
        idx = [i for i, p in enumerate(poses) if p is not None]
        if idx:
            try:
                sim, steps, settled = simulate([items[i] for i in idx], [poses[i] for i in idx], cfg, rng,
                                               max_time=min(cfg.sim_time, 1.5))
                for k, i in enumerate(idx):
                    res.poses[i] = sim[k]
                res.sim_steps, res.settled = steps, settled
            except Exception as e:
                res.note = f"physics settle failed ({e}); parts keep their geometric resting poses"
    return res


def _place_physics(items, cfg, rng, use_bin: bool) -> PlacementResult:
    if use_bin:
        sx, sy, h = cfg.bin_size
        region = (sx * 0.8, sy * 0.8)
        bin_box = (sx, sy, h, cfg.bin_wall_thickness)
    else:
        region = tuple(cfg.area_size)
        bin_box = None
    cr = cfg.cluster_radius.sample(rng)
    cx = rng.uniform(-max(0.0, region[0] / 2 - cr), max(0.0, region[0] / 2 - cr))
    cy = rng.uniform(-max(0.0, region[1] / 2 - cr), max(0.0, region[1] / 2 - cr))
    if use_bin:
        cx = cy = 0.0
        cr = min(cr, min(region) / 2)
    init, spheres = [], []
    order = rng.permutation(len(items))
    base_z = (cfg.bin_size[2] if use_bin else 0.0)
    for k in order:
        it = items[k]
        mode = {"random": "uniform_so3", "stable_pose": "stable_pose", "upright": "upright"}[cfg.initial_orientation]
        R = _sample_orientation(it, mode, rng, cfg.yaw, cfg.tilt_jitter_deg)
        r = _bound_radius(it)
        ang, rad = rng.uniform(0, 2 * math.pi), cr * math.sqrt(rng.uniform())
        p = np.array([cx + rad * math.cos(ang), cy + rad * math.sin(ang), base_z + r + cfg.drop_height.sample(rng)])
        if not cfg.sequential_drop:
            # lift until no bounding-sphere overlap with already placed items
            for _ in range(500):
                if all(np.linalg.norm(p - q) > r + rq + 1e-3 for q, rq in spheres):
                    break
                p[2] += r * 0.5
        spheres.append((p.copy(), r))
        init.append((k, make_T(R, p - R @ _hull_center(it))))
    init.sort(key=lambda x: x[0])
    init_poses = [T for _, T in init]
    release = None
    if cfg.sequential_drop:
        rank = {int(k): j for j, k in enumerate(order)}
        # one part every 0.2 s: enough to land (10 cm fall = 0.14 s) before the next one arrives. (0.6 s made a
        # 30-part pile simulate 18 s, i.e. ~15 s of computing per scene.)
        release = [SEQUENTIAL_DROP_INTERVAL * rank[i] for i in range(len(items))]
    max_time = cfg.sim_time + (max(release) if release else 0.0)
    try:
        poses, steps, settled = simulate(items, init_poses, cfg, rng, bin_box=bin_box, release_times=release,
                                         max_time=max_time)
    except Exception as e:
        # fall back to separated stable placement - and say so (it is not what was asked for)
        res = _place_separated(items, cfg, rng)
        res.note = f"physics simulation failed ({e}); parts were placed separately in stable poses instead"
        return res
    out: List[Optional[np.ndarray]] = list(poses)
    if cfg.remove_out_of_area:
        lim = np.asarray(cfg.bin_size[:2] if use_bin else cfg.area_size, float) / 2 * 1.15 + 0.01
        for i, T in enumerate(out):
            if T is None:
                continue
            c = T[:3, :3] @ _hull_center(items[i]) + T[:3, 3]     # where the part is (not its CAD origin)
            r = _bound_radius(items[i])
            if abs(c[0]) > lim[0] + r or abs(c[1]) > lim[1] + r or c[2] < -0.05:
                out[i] = None
    return PlacementResult(out, bin_box=bin_box, sim_steps=steps, settled=settled)
