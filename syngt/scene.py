"""Scene sampling and multi-camera frame rendering (shared by preview and generation)."""
from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from . import textures as tx
from .annotations import compute_annotations
from .assets import MeshAsset, asset_from_trimesh, list_mesh_files, load_asset, make_primitive_asset
from .config import MaterialConfig, ProjectConfig
from .geometry import transform_points
from .imaging import CameraCanvas, apply_depth_noise, apply_jpeg, apply_sensor_cpu, sample_sensor_params
from .placement import Item, place
from .renderer import Ground, Instance, Light, Renderer, RenderScene
from .viewpoints import sample_scene_views

GREY_CARD = 0.18 / 0.9   # radiance of an 18 % grey card relative to white_radiance() (albedo 0.9)
PRIMITIVE_POOL = 8   # aspect-ratio variants per primitive distractor shape (built once, scaled per instance)

DISTRACTOR_ID0 = 1000
BIN_ID = 2000


def kelvin_to_rgb(k: float) -> np.ndarray:
    t = k / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1) ** 2.2
    return c / max(c.max(), 1e-6)


def effective_material_mode(mode: str, asset_has_colors: bool) -> str:
    """'original' keeps the file's texture / colours. A file without any (STL, STEP ...) has nothing to keep, so
    the model's own colour is used - a colour picked in the UI is never silently ignored."""
    return "fixed_color" if mode == "original" and not asset_has_colors else mode


def _material(m: MaterialConfig, rng: np.random.Generator, asset_has_colors: bool = True) -> dict:
    rough, metal = m.roughness.sample(rng), m.metallic.sample(rng)
    mode = effective_material_mode(m.mode, asset_has_colors)
    if mode == "random_color":
        rgb = tx.random_color(rng, (0.0, max(0.0, min(1.0, m.random_saturation_max))), (0.15, 1.0))
        return dict(base_color=tuple(np.power(rgb, 2.2)), roughness=rough, metallic=metal)
    if mode in ("fixed_color", "palette"):
        base = m.color
        if mode == "palette" and m.palette:
            base = m.palette[int(rng.integers(len(m.palette)))]
        h, s, v = colorsys.rgb_to_hsv(*[float(np.clip(c, 0, 1)) for c in base[:3]])
        h = (h + rng.uniform(-m.hue_jitter, m.hue_jitter)) % 1.0
        s = float(np.clip(s * (1 + rng.uniform(-m.saturation_jitter, m.saturation_jitter)), 0, 1))
        v = float(np.clip(v * (1 + rng.uniform(-m.value_jitter, m.value_jitter)), 0, 1))
        rgb = np.array(colorsys.hsv_to_rgb(h, s, v))
        return dict(base_color=tuple(np.power(rgb, 2.2)), roughness=rough, metallic=metal)
    vj = 1 + rng.uniform(-m.value_jitter, m.value_jitter)
    tint = np.full(3, vj)
    if m.hue_jitter > 0 or m.saturation_jitter > 0:
        tint = tint * (1 + rng.uniform(-max(m.hue_jitter, m.saturation_jitter), max(m.hue_jitter, m.saturation_jitter), 3))
    return dict(base_color=None, tint=tuple(np.clip(tint, 0, 3)), roughness=rough, metallic=metal)


def sample_lights(lc, rng, center) -> Tuple[List[Light], np.ndarray, np.ndarray]:
    """One random lighting set-up for a LightingConfig: (lights, sky ambient, ground ambient)."""
    lights = []
    for i in range(max(0, lc.num_lights.sample(rng))):
        kind = lc.light_type if lc.light_type != "mixed" else str(rng.choice(["point", "directional"]))
        az, el = rng.uniform(0, 2 * math.pi), math.radians(lc.elevation_deg.sample(rng))
        d = lc.distance.sample(rng)
        direction = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        col = kelvin_to_rgb(lc.color_temperature.sample(rng)) * lc.intensity.sample(rng)
        if kind == "point":
            lights.append(Light("point", center + direction * d, col * d * d))
        else:
            lights.append(Light("directional", direction, col))
    amb = lc.ambient.sample(rng)
    sky = kelvin_to_rgb(rng.uniform(5500, 9000)) * amb
    grd = kelvin_to_rgb(rng.uniform(3500, 6000)) * amb * 0.35
    return lights, sky, grd


def white_radiance(lights, amb_sky, center) -> float:
    """Linear radiance of an upward-facing white (albedo 0.9) diffuse surface at the scene centre."""
    e = 0.0
    for L in lights:
        col = float(np.mean(L.color))
        if L.kind == "directional":
            d = np.asarray(L.vec, float)
            d = d / (np.linalg.norm(d) + 1e-9)
            e += col * max(d[2], 0.0)
        else:
            v = np.asarray(L.vec, float) - np.asarray(center, float)
            dist = np.linalg.norm(v) + 1e-9
            e += col / dist ** 2 * max(v[2] / dist, 0.0)
    return 0.9 * (e / np.pi + float(np.mean(amb_sky)))


HIGHLIGHT_LIMIT = 0.92      # brightest lit part surface on screen (0..1): above this, shading detail is lost


def light_metered_exposure(white: float, target: float, tonemap: str, contrast: float, instances) -> Tuple[float, bool]:
    """Exposure gain for auto exposure that meters the scene light: an 18 % grey card is shown at `target`.
    Like a real camera it protects highlights - the lit top face of the brightest part stays below
    HIGHLIGHT_LIMIT, so white parts keep their shading instead of clipping to pure white. Both depend only on
    the lighting and the parts, never on the table or background. Returns (gain, highlight limit was applied)."""
    from .imaging import inverse_display
    gain = inverse_display(target, tonemap) / max(white * GREY_CARD, 1e-6)
    # albedo of the brightest annotated part (linear); file colours / textures are unknown here -> assume white
    albedo = max([min(1.0, max(i.base_color)) if i.base_color is not None else 0.9
                  for i in instances if i.annotate] or [0.9])
    y_max = 0.5 + (HIGHLIGHT_LIMIT - 0.5) / max(contrast, 1e-3)      # before the contrast step that follows
    limit = inverse_display(y_max, tonemap) / max(white * albedo / 0.9, 1e-6)
    return (limit, True) if gain > limit else (gain, False)


@dataclass
class SceneState:
    scene_idx: int
    instances: List[Instance]
    annotated: List[dict]                  # inst_id, obj_id, class_name, model_name, asset, T_world, scale, model_idx
    ground: Ground
    lights: List[Light]
    amb_sky: np.ndarray
    amb_ground: np.ndarray
    center: np.ndarray
    radius: float
    views: List[List[np.ndarray]]          # [view][cam] T_world_cam
    info: dict = field(default_factory=dict)


@dataclass
class CameraFrame:
    cam_name: str
    cam_idx: int
    view_idx: int
    rgb: np.ndarray                        # uint8 HxWx3 (or HxWx4 for transparent bg)
    depth: Optional[np.ndarray]            # float32 metres, sensor model applied
    depth_gt: np.ndarray                   # float32 metres, ideal
    inst: np.ndarray                       # int32 instance ids (objects only)
    semantic: np.ndarray                   # int32 class ids
    normals: Optional[np.ndarray]
    K: np.ndarray
    dist: List[float]
    T_world_cam: np.ndarray
    annotations: List[dict]
    sensor_params: dict
    background_mode: str


class SceneSampler:
    def __init__(self, cfg: ProjectConfig, renderer: Optional[Renderer] = None,
                 log: Callable[[str], None] = lambda s: None):
        self.cfg = cfg
        self.log = log
        self._renderer = renderer
        self.models = cfg.enabled_models()
        if not cfg.cameras:
            raise ValueError("No camera is configured (Cameras tab)")
        self.assets: List[MeshAsset] = []
        self.warnings: List[str] = []
        pc = cfg.placement
        for m in self.models:
            t0 = time.time()
            a = load_asset(m.path, m.unit, m.recenter, cfg.render.smooth_angle_deg, m.class_name,
                           compute_stable=True, collision=pc.collision_geometry, max_parts=pc.max_decomposition_parts,
                           stable_max_poses=pc.stable_max_poses, stable_min_prob=pc.stable_min_prob)
            self.assets.append(a)
            ext = float(a.extent.max())
            if ext < 0.002 or ext > 2.0:
                self.warnings.append(f"'{a.name}' is {ext * 1000:.3g} mm large with unit '{m.unit}' "
                                     f"(file looks like '{a.detected_unit}') - check the unit")
            self.log(f"loaded {a.name}: {a.n_faces} faces, extent {np.round(a.extent * 1000, 1)} mm, "
                     f"{len(a.stable_poses)} stable poses ({time.time() - t0:.1f}s)")
        self.distractor_files = list_mesh_files(cfg.objects.distractors.mesh_dir) if cfg.objects.distractors.enabled else []
        self._distractor_cache: Dict[str, MeshAsset] = {}
        self._primitive_cache: Dict[str, MeshAsset] = {}
        self.ground_images = tx.list_images(cfg.scene.ground.texture_dir)
        self.bg_images = tx.list_images(cfg.scene.background.image_dir)
        # class names / unique BOP ids
        cids = [m.class_id for m in self.models]
        self.unique_class_ids = len(set(cids)) == len(cids)
        self.class_names: Dict[int, str] = {}
        for m, a in zip(self.models, self.assets):
            self.class_names.setdefault(m.class_id, m.class_name)

    @property
    def renderer(self) -> Renderer:
        if self._renderer is None:
            self._renderer = Renderer(self.cfg.render.msaa_samples, self.cfg.scene.lighting.shadow_map_size)
        return self._renderer

    def release_scene_resources(self) -> None:
        """Free GPU objects that only the last scene needed (keeps the models and the distractor pool loaded)."""
        keep = ({a.key for a in self.assets} | {a.key for a in self._primitive_cache.values()}
                | {a.key for a in self._distractor_cache.values()})
        self.renderer.release_unused(keep)

    def bop_obj_id(self, model_idx: int) -> int:
        return self.models[model_idx].class_id if self.unique_class_ids else model_idx + 1

    # ------------------------------------------------------------------ scene
    def _choose_instances(self, rng) -> List[int]:
        oc = self.cfg.objects
        n_models = len(self.models)
        if n_models == 0:
            return []
        if oc.count_mode == "per_model":
            out = []
            for i, m in enumerate(self.models):
                out += [i] * m.count.sample(rng)
            if not out and any(max(m.count.min, m.count.max) > 0 for m in self.models):
                out = [int(rng.integers(n_models))]     # the ranges allow parts, this draw gave none: place one
            return out                                  # all counts fixed at 0 -> a background-only scene
        n = oc.total_count.sample(rng)
        w = np.array([max(m.weight, 0.0) for m in self.models], float)
        w = w / w.sum() if w.sum() > 0 else np.full(n_models, 1 / n_models)
        k = min(max(1, oc.min_classes_per_scene), n_models, n, int((w > 0).sum()))   # weight 0 = never drawn
        forced = list(rng.choice(n_models, size=k, replace=False, p=w))
        rest = list(rng.choice(n_models, size=max(0, n - k), replace=True, p=w))
        return [int(x) for x in forced + rest]

    def _distractor_assets(self, rng) -> List[MeshAsset]:
        dc = self.cfg.objects.distractors
        if not dc.enabled:
            return []
        out = []
        for i in range(dc.count.sample(rng)):
            size = dc.size.sample(rng)
            if self.distractor_files and (not dc.shapes or rng.random() < 0.5):
                f = self.distractor_files[int(rng.integers(len(self.distractor_files)))]
                if f not in self._distractor_cache:
                    try:
                        self._distractor_cache[f] = load_asset(f, dc.mesh_unit, False, self.cfg.render.smooth_angle_deg,
                                                               compute_stable=True)
                    except Exception as e:
                        self.log(f"distractor {f} failed: {e}")
                        continue
                out.append((self._distractor_cache[f], size))
            elif dc.shapes:
                shape = str(rng.choice(dc.shapes))
                out.append((self._primitive(shape, int(rng.integers(PRIMITIVE_POOL))), size))
        return out

    def _primitive(self, shape: str, variant: int) -> MeshAsset:
        """Unit-size primitive (largest dim <= 1 m) from a small per-shape pool, scaled per instance. Building one
        (convex hull, stable poses, collision) takes ~0.5 s, so they are made once instead of every scene."""
        key = f"{shape}#{variant}"
        a = self._primitive_cache.get(key)
        if a is None:
            seed = [variant, sum(map(ord, shape))]
            a = self._primitive_cache[key] = make_primitive_asset(shape, 1.0, np.random.default_rng(seed), variant)
        return a

    def _ground(self, rng) -> Ground:
        g = self.cfg.scene.ground
        ground = Ground(enabled=g.enabled, visible=g.enabled and rng.random() < g.visible_probability,
                        extent=g.extent_m, roughness=g.roughness.sample(rng), rotation_deg=g.rotation_deg.sample(rng))
        mode = g.texture_mode
        if mode == "mixed":
            opts = ["solid_random", "procedural"] + (["image_folder"] * 2 if self.ground_images else [])                 + (["shadow_catcher"] * 2 if self.bg_images else [])                 + (["palette"] * 3 if g.palette else [])               # measured real table colours
            mode = str(rng.choice(opts))
        if mode == "image_folder" and not self.ground_images:
            mode = "procedural"
        if mode == "palette" and not g.palette:
            mode = "procedural"
        if mode == "palette":           # real table colour (+ subtle texture so it is not perfectly flat)
            c = np.asarray(g.palette[int(rng.integers(len(g.palette)))], float)
            c = np.clip(c * rng.uniform(0.93, 1.07), 0, 1)
            seed = int(rng.integers(0, 2**31 - 1))
            n = tx.procedural_texture(np.random.default_rng(seed), 512).astype(np.float32).mean(2, keepdims=True) / 255.0
            amp = rng.uniform(0.02, 0.08)
            ground.texture = np.clip((c[None, None] * (1 - amp + 2 * amp * n)) * 255, 0, 255).astype(np.uint8)
            ground.key = f"pal_{seed}"
            ground.color = (1.0, 1.0, 1.0)
            ground.tile_size = g.texture_size_m.sample(rng)
        elif mode == "shadow_catcher":   # invisible plane over the background photo, only shadows
            ground.catcher = True
            ground.catcher_strength = g.catcher_strength.sample(rng)
            ground.texture = None
        elif mode == "solid_random":
            ground.color = tuple(np.power(tx.random_color(rng, (0, max(0.0, min(1.0, g.saturation_max))),
                                                          (0.1, 0.95)), 2.2))
            ground.texture = None
        elif mode == "procedural":
            seed = int(rng.integers(0, 2**31 - 1))
            ground.texture = tx.desaturate(tx.procedural_texture(np.random.default_rng(seed), 1024), g.saturation_max)
            ground.key = f"proc_{seed}_{g.saturation_max:.2f}"
            ground.color = (1.0, 1.0, 1.0)
            ground.tile_size = g.texture_size_m.sample(rng)
        else:
            f = self.ground_images[int(rng.integers(len(self.ground_images)))]
            img = tx.load_image_rgb(f, 4096)
            if rng.random() < 0.5:
                seed = int(rng.integers(0, 2**31 - 1))
                img = tx.augment_texture(img, np.random.default_rng(seed))
                ground.key = f"{f}|{tx.file_stamp(f)}|{seed}"
            else:
                ground.key = f"{f}|{tx.file_stamp(f)}"
            ground.texture = img
            ground.color = (1.0, 1.0, 1.0)
            size = tx.texture_physical_size(f)
            ground.tile_size = size[0] if size else g.texture_size_m.sample(rng)
            ground.tile_size_y = size[1] if size else ground.tile_size * img.shape[0] / img.shape[1]
        ground.mode = mode  # type: ignore[attr-defined]
        return ground

    def _lights(self, rng, center) -> Tuple[List[Light], np.ndarray, np.ndarray]:
        return sample_lights(self.cfg.scene.lighting, rng, center)



    def sample_scene(self, scene_idx: int, rng: Optional[np.random.Generator] = None) -> SceneState:
        cfg = self.cfg
        rng = rng or np.random.default_rng([cfg.generation.seed, scene_idx])
        chosen = self._choose_instances(rng)
        items, meta = [], []
        for mi in chosen:
            m, a = self.models[mi], self.assets[mi]
            s = m.scale_jitter.sample(rng)
            items.append(Item(a, s, m.friction, m.density))
            meta.append(("obj", mi, s))
        for a, s in self._distractor_assets(rng):
            if a.path:  # mesh from folder: scale to desired size
                s = s / max(a.diameter, 1e-6)
            items.append(Item(a, s, 0.6, 800.0))
            meta.append(("distractor", -1, s))
        t0 = time.time()
        pres = place(items, cfg.placement, rng, floor=cfg.scene.ground.enabled)
        if pres.note and pres.note not in self.warnings:
            self.warnings.append(pres.note)
        t_place = time.time() - t0

        instances: List[Instance] = []
        annotated: List[dict] = []
        k_obj, k_dis = 1, DISTRACTOR_ID0
        for (kind, mi, s), it, T in zip(meta, items, pres.poses):
            if T is None:
                continue
            if kind == "obj":
                mat = _material(self.models[mi].material, rng, it.asset.has_own_colors)
                iid = k_obj
                k_obj += 1
                annotated.append(dict(inst_id=iid, obj_id=self.models[mi].class_id, bop_obj_id=self.bop_obj_id(mi),
                                      class_name=self.class_names[self.models[mi].class_id],
                                      model_name=self.assets[mi].name, model_idx=mi, asset=it.asset, T_world=T, scale=s))
            else:
                mat = _material(cfg.objects.distractors.material, rng, it.asset.has_own_colors)
                iid = k_dis
                k_dis += 1
            instances.append(Instance(it.asset, T, iid, annotate=(kind == "obj"), base_color=mat["base_color"],
                                      tint=mat.get("tint", (1, 1, 1)), roughness=mat["roughness"],
                                      metallic=mat["metallic"], scale=s))
        if pres.bin_box is not None and cfg.placement.render_bin:
            instances.append(self._bin_instance(pres.bin_box, rng))

        # scene extent from annotated objects (fallback: all)
        pts = []
        for inst in instances:
            if inst.annotate or not annotated:
                pts.append(transform_points(inst.T_world, inst.asset.bbox_corners() * inst.scale))
        if pts:
            P = np.vstack(pts)
            center = 0.5 * (P.min(0) + P.max(0))
            radius = float(max(0.03, 0.5 * np.linalg.norm(P.max(0) - P.min(0))))
        else:
            center, radius = np.zeros(3), 0.1
        ground = self._ground(rng)
        lights, sky, grd = self._lights(rng, center)
        centers = np.array([transform_points(a["T_world"], (0.5 * (a["asset"].bbox_min + a["asset"].bbox_max) * a["scale"])[None])[0]
                            for a in annotated]) if annotated else np.zeros((0, 3))
        views = sample_scene_views(cfg.viewpoints, cfg.cameras, center, radius, centers, rng)
        info = dict(placement_mode=cfg.placement.mode, sim_steps=pres.sim_steps, settled=pres.settled,
                    placement_time_s=round(t_place, 3), n_requested=len([m for m in meta if m[0] == "obj"]),
                    n_placed=len(annotated), ground_mode=getattr(ground, "mode", ""), placement_note=pres.note)
        return SceneState(scene_idx, instances, annotated, ground, lights, sky, grd, center, radius, views, info)

    def _bin_instance(self, bin_box, rng) -> Instance:
        sx, sy, h, w = bin_box
        parts = []
        for pos, ext in (((-(sx + w) / 2, 0, h / 2), (w, sy + 2 * w, h)), (((sx + w) / 2, 0, h / 2), (w, sy + 2 * w, h)),
                         ((0, -(sy + w) / 2, h / 2), (sx + 2 * w, w, h)), ((0, (sy + w) / 2, h / 2), (sx + 2 * w, w, h)),
                         ((0, 0, -0.0004), (sx + 2 * w, sy + 2 * w, 0.001))):   # top 0.1 mm above the table: no z-fighting
            b = trimesh.creation.box(extents=ext)
            b.apply_translation(pos)
            parts.append(b)
        m = trimesh.util.concatenate(parts)
        a = asset_from_trimesh(m, f"bin_{sx:.3f}_{sy:.3f}_{h:.3f}", stable=False)
        a.key = f"bin_{sx:.4f}_{sy:.4f}_{h:.4f}_{w:.4f}"
        rgb = tx.random_color(rng, (0, 0.8), (0.2, 0.9))
        return Instance(a, np.eye(4), BIN_ID, annotate=False, base_color=tuple(np.power(rgb, 2.2)),
                        roughness=rng.uniform(0.4, 0.9), metallic=0.0)

    # ------------------------------------------------------------------ render
    def render_view(self, st: SceneState, view_idx: int, rng: np.random.Generator,
                    want_normals: bool = False, want_full_masks: bool = False) -> List[CameraFrame]:
        cfg = self.cfg
        lights, sky, grd = st.lights, st.amb_sky, st.amb_ground
        if cfg.scene.lighting.relight_per_view and view_idx > 0:
            lights, sky, grd = self._lights(rng, st.center)
        rs = RenderScene(st.instances, st.ground, lights, sky, grd, shadows=cfg.scene.lighting.shadows,
                         center=st.center, radius=st.radius,
                         shadow_softness=cfg.scene.lighting.shadow_softness.sample(rng))
        frames = []
        obj_ids = {a["inst_id"]: a["obj_id"] for a in st.annotated}
        for ci, cam in enumerate(cfg.cameras):
            T_wc = st.views[view_idx][ci]
            K = cam.K()
            if cam.intrinsics_jitter > 0:
                j = 1 + rng.uniform(-cam.intrinsics_jitter, cam.intrinsics_jitter, 4)
                K = K.copy()
                K[0, 0] *= j[0]
                K[1, 1] *= j[1]
                K[0, 2] = cam.cx + (j[2] - 1) * cam.width * 0.5
                K[1, 2] = cam.cy + (j[3] - 1) * cam.height * 0.5
            canvas = CameraCanvas(K, cam.width, cam.height, cam.dist, cfg.render.canvas_padding)
            gpu_pp = cfg.render.gpu_postprocess
            out = self.renderer.render(rs, canvas.Kr, canvas.Wr, canvas.Hr, T_wc, cfg.render.near, cfg.render.far,
                                       solo_ids=[a["inst_id"] for a in st.annotated],
                                       read_color=not gpu_pp, read_normals=want_normals)
            inst_full = canvas.to_image(out.inst, nearest=True)
            depth_gt = canvas.to_image(out.depth, nearest=True)
            normals = canvas.to_image(out.normals, nearest=True) if want_normals else None
            bg_u8, bg_mode = tx.make_background_u8(cfg.scene.background.mode, cam.width, cam.height, rng,
                                                   self.bg_images, cfg.scene.background.random_crop,
                                                   cfg.scene.background.flip, cfg.scene.background.saturation_max,
                                                   (cfg.scene.background.crop_scale.min, cfg.scene.background.crop_scale.max))
            bg_gain = cfg.scene.background.brightness.sample(rng)
            sp = sample_sensor_params(cfg.sensor, rng)
            # real photos already carry the camera's look -> composite them AFTER the simulated sensor
            # the photo is the real floor only with the invisible (shadow-catching) table or no table;
            # a photo far behind an opaque synthetic table is just a backdrop -> normal auto exposure
            photo = (bg_mode == "image_folder" and cfg.scene.background.photos_are_camera_images
                     and (st.ground.catcher or not (st.ground.enabled and st.ground.visible)))
            if photo and sp.get("enabled", True):
                from .imaging import photo_exposure
                sp["exposure"] = photo_exposure(bg_u8, white_radiance(lights, sky, st.center),
                                                sp.get("tonemap", "filmic"))
                sp["photo_exposure"] = True
            elif sp.get("enabled", True) and sp.get("ae") and cfg.sensor.ae_metering == "light":
                sp["exposure"], sp["ae_highlight_limited"] = light_metered_exposure(
                    white_radiance(lights, sky, st.center), float(sp["ae_target"]), sp.get("tonemap", "filmic"),
                    float(sp.get("contrast", 1.0)), st.instances)
                sp["ae"] = False
            if gpu_pp:   # sensor model on the GPU, only the final 8-bit image is read back
                img = self.renderer.postprocess(canvas, bg_u8, bg_gain, sp, keep_alpha=bg_u8 is None,
                                                bg_is_photo=photo)
            else:        # CPU reference path
                rgba = canvas.to_image(out.rgba)
                alpha = np.clip(rgba[..., 3:4], 0, 1)
                lin = rgba[..., :3]
                if photo:
                    img = apply_sensor_cpu(lin, sp, alpha=alpha,
                                           bg_display=np.clip(bg_u8.astype(np.float32) / 255.0 * bg_gain, 0, 1))
                elif bg_u8 is None:     # transparent: RGBA, straight colour, alpha blurred like the colour
                    img = apply_sensor_cpu(lin, sp, alpha=alpha)
                else:
                    img = apply_sensor_cpu(lin + tx.background_to_linear(bg_u8, bg_gain) * (1 - alpha), sp)
            img = apply_jpeg(img, sp.get("jpeg_quality", 100))
            sp = {k: v for k, v in sp.items() if k not in ("enabled",)}
            depth = apply_depth_noise(depth_gt, cam.depth, K[0, 0], rng) if cam.depth.enabled else None
            anns = compute_annotations(st.annotated, canvas, out.inst, out.solo, inst_full, depth_gt, T_wc, K,
                                       cam.dist if canvas.distorted else None, cfg.output.polygon_tolerance_px,
                                       cfg.output.keypoints, want_rle=True, want_full_masks=want_full_masks)
            inst_obj = np.where(inst_full < DISTRACTOR_ID0, inst_full, 0).astype(np.int32)
            lut = np.zeros(max(obj_ids.keys(), default=0) + 1, np.int32)
            for k, v in obj_ids.items():
                lut[k] = v
            semantic = lut[np.clip(inst_obj, 0, len(lut) - 1)]
            frames.append(CameraFrame(cam.name, ci, view_idx, img, depth, depth_gt, inst_obj, semantic, normals, K,
                                      list(cam.dist), T_wc, anns, sp, bg_mode))
        return frames
