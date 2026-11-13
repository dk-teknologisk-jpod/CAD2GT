"""CAD / mesh loading, unit handling, stable poses, collision geometry, BOP model export."""
from __future__ import annotations

import hashlib
import math
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001, "inch": 0.0254}
MESH_EXTS = {".stl", ".obj", ".ply", ".glb", ".gltf", ".off", ".3mf", ".dae", ".step", ".stp", ".fbx"}


@dataclass
class MeshPart:
    positions: np.ndarray            # (N,3) float32, metres, object frame
    normals: np.ndarray              # (N,3) float32
    indices: np.ndarray              # (M,3) uint32
    uvs: Optional[np.ndarray] = None  # (N,2) float32
    colors: Optional[np.ndarray] = None  # (N,4) float32 0..1
    texture: Optional[np.ndarray] = None  # (H,W,3|4) uint8
    base_color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass
class MeshAsset:
    key: str
    name: str
    path: str
    parts: List[MeshPart]
    vertices: np.ndarray             # all vertices (m) for bounds / projection
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    diameter: float
    com: np.ndarray
    detected_unit: str
    unit_scale: float
    stable_poses: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # (R 3x3, prob)
    collision_hulls: List[np.ndarray] = field(default_factory=list)              # vertex arrays (m)
    hull_vertices: Optional[np.ndarray] = None
    n_faces: int = 0
    has_texture: bool = False
    has_colors: bool = False
    trimesh_mesh: Optional[trimesh.Trimesh] = None  # merged mesh (m) for export
    stable_pending: bool = False   # stable poses not computed yet (primitives: only needed by stable placements)

    @property
    def has_own_colors(self) -> bool:
        """True when the file itself defines the look (texture, vertex/face colours or a material colour).
        Plain geometry (STL, STEP ...) has none: it is shown in the colour chosen for the model."""
        default = MeshPart.__dataclass_fields__["base_color"].default
        return any(p.texture is not None or p.colors is not None
                   or not np.allclose(p.base_color[:3], default[:3], atol=1e-3) for p in self.parts)

    def get_stable_poses(self) -> List[Tuple[np.ndarray, float]]:
        if self.stable_pending:
            self.stable_pending = False
            hull = trimesh.Trimesh(self.hull_vertices).convex_hull
            self.stable_poses = compute_stable_poses(hull, self.com, 6, 0.0)
        return self.stable_poses

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def bbox_corners(self) -> np.ndarray:
        mn, mx = self.bbox_min, self.bbox_max
        return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])

    def bbox3d_keypoints(self) -> np.ndarray:
        """9 keypoints: centre + 8 corners (order: centre, then x-major corner order)."""
        c = 0.5 * (self.bbox_min + self.bbox_max)
        return np.vstack([c[None], self.bbox_corners()])


# --------------------------------------------------------------------------- #
# formats that store their unit: the loader already returns metres, the "file unit" setting must not scale again
SELF_SCALED_EXTS = (".step", ".stp")
CACHE_VERSION = 3       # part of every cache key: bump when the loading / shading code changes


def unit_scale(path: str, unit: str, raw_extent: float) -> Tuple[str, float]:
    """(unit used, factor to metres) for a file with the given raw extent."""
    if str(path).lower().endswith(SELF_SCALED_EXTS):
        return "m", 1.0
    use = guess_unit(raw_extent) if unit == "auto" else unit
    return use, UNIT_SCALE[use]


def guess_unit(extent_max: float) -> str:
    if extent_max > 5.0:
        return "mm"
    return "m"


def guess_file_unit(path: str) -> str:
    """Cheap unit guess from the raw file extent (no processing)."""
    try:
        sc = _load_trimesh_scene(path)
        ext = float(np.max(sc.extents))
        return unit_scale(path, "auto", ext)[0]
    except Exception:
        return "auto"


_EXTENT_CACHE: Dict[Tuple[str, str, int], Optional[float]] = {}


def model_extent(path: str, unit: str = "auto") -> Optional[float]:
    """Largest dimension of a model file in metres for the given file unit (None if it cannot be read).
    Cached per file version, so it is cheap to call from checks and read-outs."""
    try:
        key = (os.path.abspath(path), unit, os.stat(path).st_mtime_ns)
    except OSError:
        return None
    if key not in _EXTENT_CACHE:
        try:
            ext = float(np.max(_load_trimesh_scene(path).extents))
            _EXTENT_CACHE[key] = ext * unit_scale(path, unit, ext)[1]
        except Exception:
            _EXTENT_CACHE[key] = None
    return _EXTENT_CACHE[key]


def clear_cache() -> None:
    """Forget all loaded meshes (memory + disk cache); they are re-read from the CAD files on next use."""
    import shutil
    _MEM.clear()
    _EXTENT_CACHE.clear()
    shutil.rmtree(_cache_dir(), ignore_errors=True)


def _cache_dir() -> Path:
    d = Path(os.environ.get("SYNGT_CACHE", Path(tempfile.gettempdir()) / "syngt_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file_key(path: str, *extra) -> str:
    st = os.stat(path)
    h = hashlib.sha1(f"v{CACHE_VERSION}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{extra}".encode()).hexdigest()
    return h[:20]


def _load_trimesh_scene(path: str) -> trimesh.Scene:
    ext = Path(path).suffix.lower()
    if ext in (".step", ".stp"):
        try:
            import cascadio  # noqa: F401  (registers STEP loader in trimesh)
        except ImportError as e:
            raise RuntimeError("STEP import needs 'cascadio' (pip install cascadio)") from e
    if ext in (".igs", ".iges"):
        raise RuntimeError("IGES is not supported - export STEP/STL/OBJ/GLB from your CAD tool")
    loaded = trimesh.load(path, force="scene", process=False)
    return loaded


def _texture_from_material(mat) -> Tuple[Optional[np.ndarray], Tuple[float, float, float, float]]:
    img, color = None, (0.7, 0.7, 0.7, 1.0)
    if mat is None:
        return img, color
    tex = getattr(mat, "baseColorTexture", None)
    if tex is None:
        tex = getattr(mat, "image", None)
    if tex is not None:
        try:
            img = np.asarray(tex.convert("RGBA"))
        except Exception:
            img = None
    bc = getattr(mat, "baseColorFactor", None)
    if bc is None:
        bc = getattr(mat, "main_color", None)
    if bc is not None:
        bc = np.asarray(bc, dtype=np.float64).reshape(-1)
        if bc.max() > 1.0:
            bc = bc / 255.0
        if bc.size == 3:
            bc = np.append(bc, 1.0)
        color = tuple(float(x) for x in bc[:4])
    return img, color


def _mesh_to_parts(mesh: trimesh.Trimesh, smooth_angle_deg: float) -> List[MeshPart]:
    parts: List[MeshPart] = []
    visual = mesh.visual
    kind = getattr(visual, "kind", None)
    if kind == "texture" and getattr(visual, "uv", None) is not None and len(visual.uv) == len(mesh.vertices):
        img, color = _texture_from_material(getattr(visual, "material", None))
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        uv = np.asarray(visual.uv, dtype=np.float32)
        parts.append(MeshPart(np.asarray(mesh.vertices, np.float32), normals,
                              np.asarray(mesh.faces, np.uint32), uvs=uv, texture=img, base_color=color))
        return parts

    if kind == "vertex":
        vc = np.asarray(visual.vertex_colors, dtype=np.float32) / 255.0
        parts.append(MeshPart(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.vertex_normals, np.float32),
                              np.asarray(mesh.faces, np.uint32), colors=vc))
        return parts

    base = (0.7, 0.7, 0.7, 1.0)
    face_colors = None
    if kind == "face":
        fc = np.asarray(visual.face_colors, dtype=np.float32) / 255.0
        if np.allclose(fc, fc[0], atol=1e-3):
            base = tuple(float(x) for x in fc[0])
        else:
            face_colors = fc
    elif kind == "texture":  # texture visual without usable uv -> material colour only
        _, base = _texture_from_material(getattr(visual, "material", None))

    # CAD meshes: split vertices at sharp creases for clean shading
    nf = len(mesh.faces)
    if face_colors is None and nf < 400_000 and smooth_angle_deg > 0:
        try:
            mesh = mesh.copy()
            mesh.merge_vertices()       # STL = triangle soup: weld it, or every face stays flat
            sm = trimesh.graph.smooth_shade(mesh, angle=math.radians(smooth_angle_deg),
                                            facet_minarea=None)
            parts.append(MeshPart(np.asarray(sm.vertices, np.float32), np.asarray(sm.vertex_normals, np.float32),
                                  np.asarray(sm.faces, np.uint32), base_color=base))
            return parts
        except Exception:
            pass
    # flat shading fallback (also used for per-face colours)
    tri = np.asarray(mesh.vertices, np.float32)[np.asarray(mesh.faces)]
    pos = tri.reshape(-1, 3)
    fn = np.asarray(mesh.face_normals, np.float32)
    nrm = np.repeat(fn, 3, axis=0)
    idx = np.arange(len(pos), dtype=np.uint32).reshape(-1, 3)
    cols = np.repeat(face_colors, 3, axis=0) if face_colors is not None else None
    parts.append(MeshPart(pos, nrm, idx, colors=cols, base_color=base))
    return parts


def _named(asset: MeshAsset, name: str) -> MeshAsset:
    """The cached asset under the caller's name. A file used by two models (variants, colours) is loaded once;
    each model gets its own light copy so one name never overwrites the other."""
    if not name or name == asset.name:
        return asset
    import dataclasses
    return dataclasses.replace(asset, name=name)


def load_asset(path: str, unit: str = "auto", recenter: bool = False, smooth_angle_deg: float = 30.0,
               name: str = "", compute_stable: bool = True, collision: str = "convex_hull",
               max_parts: int = 16, stable_max_poses: int = 8, stable_min_prob: float = 0.0) -> MeshAsset:
    """Load a CAD/mesh file into a render+physics ready asset (metres)."""
    path = str(path)
    key = _file_key(path, unit, recenter, smooth_angle_deg, collision, max_parts, stable_max_poses, stable_min_prob)
    if key in _MEM:
        return _named(_MEM[key], name)
    cache_file = _cache_dir() / f"{key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                asset = pickle.load(f)
            _mem_put(key, asset)
            return _named(asset, name)
        except Exception:
            pass

    scene = _load_trimesh_scene(path)
    meshes = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        raise RuntimeError(f"No triangle geometry in {path}")
    allv = np.vstack([m.vertices for m in meshes])
    ext_raw = float((allv.max(0) - allv.min(0)).max())
    detected = guess_unit(ext_raw)
    use_unit, s = unit_scale(path, unit, ext_raw)

    merged = trimesh.util.concatenate([trimesh.Trimesh(m.vertices, m.faces, process=False) for m in meshes])
    merged.apply_scale(s)
    offset = np.zeros(3)
    if recenter:
        offset = -0.5 * (merged.bounds[0] + merged.bounds[1])
        merged.apply_translation(offset)

    parts: List[MeshPart] = []
    for m in meshes:
        m = m.copy()
        m.apply_scale(s)
        if recenter:
            m.apply_translation(offset)
        parts.extend(_mesh_to_parts(m, smooth_angle_deg))

    verts = np.asarray(merged.vertices, dtype=np.float64)
    bmin, bmax = verts.min(0), verts.max(0)
    # hull for diameter / stable poses / collisions
    try:
        hull = merged.convex_hull
    except Exception:
        hull = trimesh.points.PointCloud(verts).convex_hull
    hv = np.asarray(hull.vertices)
    if len(hv) > 2000:
        sel = np.random.default_rng(0).choice(len(hv), 2000, replace=False)
        hv_d = hv[sel]
    else:
        hv_d = hv
    diam = float(np.max(np.linalg.norm(hv_d[:, None] - hv_d[None], axis=-1))) if len(hv_d) < 2500 else float(np.linalg.norm(bmax - bmin))
    try:
        merged_proc = trimesh.Trimesh(merged.vertices, merged.faces, process=True)
        com = np.asarray(merged_proc.center_mass) if merged_proc.is_watertight and merged_proc.volume > 0 else np.asarray(hull.center_mass)
    except Exception:
        com = 0.5 * (bmin + bmax)

    asset = MeshAsset(key=key, name=name or Path(path).stem, path=path, parts=parts, vertices=verts.astype(np.float32),
                      bbox_min=bmin, bbox_max=bmax, diameter=diam, com=com, detected_unit=detected, unit_scale=s,
                      hull_vertices=hv.astype(np.float64), n_faces=int(len(merged.faces)),
                      has_texture=any(p.texture is not None for p in parts),
                      has_colors=any(p.colors is not None for p in parts),
                      trimesh_mesh=merged)

    if compute_stable:
        asset.stable_poses = compute_stable_poses(hull, com, stable_max_poses, stable_min_prob)
    asset.collision_hulls = compute_collision(merged, hull, collision, max_parts)

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(asset, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    _mem_put(key, asset)
    return asset


_MEM: Dict[str, MeshAsset] = {}


def _mem_put(key: str, asset: MeshAsset, max_items: int = 64) -> None:
    if len(_MEM) >= max_items:
        _MEM.pop(next(iter(_MEM)))
    _MEM[key] = asset


def compute_stable_poses(hull: trimesh.Trimesh, com: np.ndarray, max_poses: int = 8,
                         min_prob: float = 0.0) -> List[Tuple[np.ndarray, float]]:
    """Quasi-static stable resting poses (rotation only) with probabilities."""
    try:
        h = hull
        if len(h.faces) > 3000:
            h = h.simplify_quadric_decimation(face_count=3000) if hasattr(h, "simplify_quadric_decimation") else h
            h = h.convex_hull
        Ts, probs = trimesh.poses.compute_stable_poses(h, center_mass=com, sigma=0.0, n_samples=1, threshold=0.0)
    except Exception:
        Ts, probs = [], []
    out = [(np.asarray(T)[:3, :3].copy(), float(p)) for T, p in zip(Ts, probs) if p >= min_prob]
    out.sort(key=lambda x: -x[1])
    out = out[:max(1, max_poses)]
    if not out:
        out = [(np.eye(3), 1.0)]
    tot = sum(p for _, p in out)
    return [(R, p / tot) for R, p in out]


def compute_collision(mesh: trimesh.Trimesh, hull: trimesh.Trimesh, mode: str, max_parts: int) -> List[np.ndarray]:
    def _dec(vs: np.ndarray, n=64) -> np.ndarray:
        vs = np.asarray(vs, dtype=np.float64)
        if len(vs) > n:
            try:
                h = trimesh.points.PointCloud(vs).convex_hull
                if len(h.vertices) > n:
                    # farthest point subsample of hull vertices keeps shape well
                    idx = [0]
                    d = np.linalg.norm(h.vertices - h.vertices[0], axis=1)
                    for _ in range(n - 1):
                        i = int(np.argmax(d))
                        idx.append(i)
                        d = np.minimum(d, np.linalg.norm(h.vertices - h.vertices[i], axis=1))
                    return np.asarray(h.vertices)[idx]
                return np.asarray(h.vertices)
            except Exception:
                return vs[:n]
        return vs

    if mode == "convex_decomposition":
        parts = []
        try:
            import coacd
            coacd.set_log_level("error")
            m = mesh
            if len(m.faces) > 60000:
                m = m.simplify_quadric_decimation(face_count=60000) if hasattr(m, "simplify_quadric_decimation") else m
            cm = coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces))
            res = coacd.run_coacd(cm, threshold=0.05, max_convex_hull=max_parts, preprocess_mode="auto")
            parts = [_dec(v) for v, _f in res]
        except Exception:
            try:
                res = mesh.convex_decomposition()
                res = res if isinstance(res, list) else [res]
                parts = [_dec(r.vertices if hasattr(r, "vertices") else r["vertices"]) for r in res][:max_parts]
            except Exception:
                parts = []
        parts = [p for p in parts if len(p) >= 4]
        if parts:
            return parts
    return [_dec(hull.vertices, 128)]


# --------------------------------------------------------------------------- #
# Procedural distractors
# --------------------------------------------------------------------------- #
def make_primitive_asset(shape: str, size: float, rng: np.random.Generator, idx: int) -> MeshAsset:
    a = rng.uniform(0.3, 1.0, 3) * size
    if shape == "box":
        m = trimesh.creation.box(extents=a)
    elif shape == "cylinder":
        m = trimesh.creation.cylinder(radius=a[0] / 2, height=a[2], sections=40)
    elif shape == "sphere":
        m = trimesh.creation.icosphere(subdivisions=3, radius=size / 2)
    elif shape == "capsule":
        m = trimesh.creation.capsule(height=a[2], radius=min(a[0], a[1]) / 2, count=[24, 24])
    elif shape == "cone":
        m = trimesh.creation.cone(radius=a[0] / 2, height=a[2], sections=40)
    elif shape == "torus":
        r = size / 2
        m = trimesh.creation.torus(major_radius=r * 0.7, minor_radius=r * 0.25)
    else:
        m = trimesh.creation.box(extents=a)
    a = asset_from_trimesh(m, f"distractor_{shape}_{idx}", stable=False)
    a.stable_pending = True
    return a


def asset_from_trimesh(m: trimesh.Trimesh, name: str, stable: bool = True, smooth_angle_deg: float = 30.0) -> MeshAsset:
    parts = _mesh_to_parts(m, smooth_angle_deg)
    v = np.asarray(m.vertices, np.float64)
    hull = m.convex_hull
    com = np.asarray(hull.center_mass)
    a = MeshAsset(key=f"prim_{name}_{id(m)}", name=name, path="", parts=parts, vertices=v.astype(np.float32),
                  bbox_min=v.min(0), bbox_max=v.max(0), diameter=float(np.linalg.norm(v.max(0) - v.min(0))),
                  com=com, detected_unit="m", unit_scale=1.0, hull_vertices=np.asarray(hull.vertices),
                  n_faces=len(m.faces), trimesh_mesh=m)
    if stable:
        a.stable_poses = compute_stable_poses(hull, com, 6, 0.0)
    a.collision_hulls = compute_collision(m, hull, "convex_hull", 1)
    return a


def export_bop_model(asset: MeshAsset, out_path: Path) -> dict:
    """Write a BOP-style model (PLY, millimetres, with normals) and return its models_info entry."""
    m = asset.trimesh_mesh.copy()
    m.apply_scale(1000.0)
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.export(out_path, file_type="ply", encoding="binary", vertex_normal=True)
    mn, mx = m.bounds
    return {
        "diameter": asset.diameter * 1000.0,
        "min_x": float(mn[0]), "min_y": float(mn[1]), "min_z": float(mn[2]),
        "size_x": float(mx[0] - mn[0]), "size_y": float(mx[1] - mn[1]), "size_z": float(mx[2] - mn[2]),
    }


def symmetry_info(kind: str, order: int, asset: MeshAsset) -> dict:
    """BOP models_info symmetry entries (offsets in mm)."""
    if kind == "none":
        return {}
    from scipy.spatial.transform import Rotation
    if kind == "box_xyz":  # cuboid-like part: 180 deg about x, y and z through the bbox centre
        c = 0.5 * (asset.bbox_min + asset.bbox_max) * 1000.0
        out = []
        for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(np.asarray(ax, float) * math.pi).as_matrix()
            T[:3, 3] = c - T[:3, :3] @ c
            out.append(T.reshape(-1).tolist())
        return {"symmetries_discrete": out}
    ax = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[kind[-1]]
    centre_mm = (0.5 * (asset.bbox_min + asset.bbox_max) * 1000.0).tolist()
    if kind.startswith("continuous"):
        off = [c if a == 0 else 0.0 for c, a in zip(centre_mm, ax)]
        return {"symmetries_continuous": [{"axis": ax, "offset": off}]}
    from scipy.spatial.transform import Rotation
    out = []
    c = np.asarray(centre_mm)
    order = max(2, int(order))      # 0 / 1 would divide by zero and write NaN matrices into models_info
    for k in range(1, order):
        Rm = Rotation.from_rotvec(np.asarray(ax, float) * 2 * math.pi * k / order).as_matrix()
        t = c - Rm @ c
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = t
        out.append(T.reshape(-1).tolist())
    return {"symmetries_discrete": out}


def list_mesh_files(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in MESH_EXTS and ".syngt_cache" not in p.parts:
            out.append(str(p))
    return out
