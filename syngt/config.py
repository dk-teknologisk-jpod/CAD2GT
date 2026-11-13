"""Configuration model for the synthetic GT generator.

Everything that can be permuted lives here. The whole project is a single
pydantic model that round-trips to YAML, so a dataset can be regenerated
bit-exactly from the config + seed that is copied into every output folder.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(validate_assignment=False, extra="ignore")


# --------------------------------------------------------------------------- #
# Random ranges
# --------------------------------------------------------------------------- #
class Range(_Base):
    """A scalar random variable. ``min == max`` means constant."""

    min: float = 0.0
    max: float = 0.0
    dist: Literal["uniform", "loguniform", "normal"] = "uniform"

    def sample(self, rng: np.random.Generator) -> float:
        lo, hi = (self.min, self.max) if self.min <= self.max else (self.max, self.min)
        if hi - lo < 1e-12:
            return float(lo)
        if self.dist == "loguniform" and lo > 0:
            return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        if self.dist == "normal":
            # mean in the middle, +-2 sigma spans the range, clipped
            mu, sd = 0.5 * (lo + hi), (hi - lo) / 4.0
            return float(np.clip(rng.normal(mu, sd), lo, hi))
        return float(rng.uniform(lo, hi))


class IntRange(_Base):
    min: int = 1
    max: int = 1

    def sample(self, rng: np.random.Generator) -> int:
        lo, hi = sorted((int(self.min), int(self.max)))
        return int(rng.integers(lo, hi + 1))


def R(a, b, dist="uniform") -> Range:
    return Range(min=a, max=b, dist=dist)


# --------------------------------------------------------------------------- #
# Objects
# --------------------------------------------------------------------------- #
class MaterialConfig(_Base):
    mode: Literal["original", "fixed_color", "random_color", "palette"] = "original"
    color: List[float] = Field(default_factory=lambda: [0.7, 0.7, 0.7])  # display RGB 0..1
    palette: List[List[float]] = Field(default_factory=list)            # mode "palette": pick one per instance
    random_saturation_max: float = 0.05                                  # mode "random_color" (real parts/clutter are rarely vivid)
    hue_jitter: float = 0.0          # +- in [0,0.5] (fraction of hue circle)
    saturation_jitter: float = 0.0   # +- multiplicative
    value_jitter: float = 0.0        # +- multiplicative
    roughness: Range = Field(default_factory=lambda: R(0.3, 0.8))
    metallic: Range = Field(default_factory=lambda: R(0.0, 0.2))


class ObjectModel(_Base):
    enabled: bool = True
    path: str = ""
    name: str = ""                    # class name in all labels; empty = the CAD file name (see class_name)
    class_id: int = 1                 # BOP obj_id / detector class id (1-based)
    unit: Literal["auto", "mm", "cm", "m", "inch"] = "auto"  # auto: >5 file units -> mm, else m
    recenter: bool = False            # move origin to bbox centre (changes GT frame!)
    scale_jitter: Range = Field(default_factory=lambda: R(1.0, 1.0))
    count: IntRange = Field(default_factory=lambda: IntRange(min=2, max=5))
    weight: float = 1.0               # sampling weight when total instance count is used
    material: MaterialConfig = Field(default_factory=MaterialConfig)
    symmetry: Literal["none", "continuous_x", "continuous_y", "continuous_z",
                      "discrete_x", "discrete_y", "discrete_z", "box_xyz"] = "none"
    symmetry_order: int = 2           # for discrete symmetry
    density: float = 1100.0           # kg/m^3 for physics (ABS-like plastic)
    friction: float = 0.6

    @property
    def cad_name(self) -> str:
        """Name of the CAD file without folder and extension (works for paths written on either OS)."""
        return self.path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] if self.path else ""

    @property
    def class_name(self) -> str:
        """The name written into every label: the CAD file name, unless another name was typed in."""
        return self.name.strip() or self.cad_name


class DistractorConfig(_Base):
    enabled: bool = True
    count: IntRange = Field(default_factory=lambda: IntRange(min=0, max=4))
    shapes: List[str] = Field(default_factory=lambda: ["box", "cylinder", "sphere", "capsule"])
    mesh_dir: str = ""                # optional folder of extra (unlabelled) meshes
    mesh_unit: Literal["auto", "mm", "cm", "m", "inch"] = "auto"
    size: Range = Field(default_factory=lambda: R(0.008, 0.04))  # metres, largest dim (similar to small parts)
    material: MaterialConfig = Field(default_factory=lambda: MaterialConfig(mode="random_color", random_saturation_max=0.05))


class ObjectsConfig(_Base):
    models: List[ObjectModel] = Field(default_factory=list)
    count_mode: Literal["per_model", "total"] = "per_model"
    total_count: IntRange = Field(default_factory=lambda: IntRange(min=3, max=8))
    min_classes_per_scene: int = 1
    distractors: DistractorConfig = Field(default_factory=DistractorConfig)


# --------------------------------------------------------------------------- #
# Placement / physics
# --------------------------------------------------------------------------- #
class PlacementConfig(_Base):
    mode: Literal["free_floating", "stable_separated", "physics_drop", "physics_bin"] = "physics_drop"
    # support area (metres) on the z=0 plane, centred on origin
    area_size: List[float] = Field(default_factory=lambda: [0.25, 0.20])
    # --- stable_separated
    stable_min_prob: float = 0.02         # ignore very unlikely stable poses
    stable_max_poses: int = 8             # keep the N most likely poses per object
    min_spacing: float = 0.005            # metres between footprints
    yaw: Range = Field(default_factory=lambda: R(-180, 180))
    tilt_jitter_deg: float = 0.0          # small tilt around stable pose
    settle_with_physics: bool = True      # short physics settle after placement
    # --- physics_drop / physics_bin
    cluster_radius: Range = Field(default_factory=lambda: R(0.015, 0.08))  # spread of drop locations (small => piles/leaning)
    drop_height: Range = Field(default_factory=lambda: R(0.02, 0.10))
    initial_orientation: Literal["random", "stable_pose", "upright"] = "random"
    sequential_drop: bool = False         # drop one by one (stacks) instead of simultaneously
    friction: Range = Field(default_factory=lambda: R(0.4, 0.9))
    restitution_damping: float = 1.0      # solref damping ratio (1 = critically damped)
    sim_time: float = 4.0                 # seconds max (stops earlier once settled)
    timestep: float = 0.001                # small/light parts need a fine step
    collision_geometry: Literal["convex_hull", "convex_decomposition"] = "convex_hull"
    max_decomposition_parts: int = 16
    bin_size: List[float] = Field(default_factory=lambda: [0.20, 0.15, 0.06])  # inner x,y,height
    bin_wall_thickness: float = 0.005
    render_bin: bool = True
    remove_out_of_area: bool = True       # drop instances that left the area/bin
    # --- free_floating
    float_volume: List[float] = Field(default_factory=lambda: [0.12, 0.12, 0.08])  # box centred at float_center
    float_center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.06])
    float_avoid_collisions: bool = True
    float_rotation: Literal["uniform_so3", "stable_pose", "upright_yaw"] = "uniform_so3"


class GroundConfig(_Base):
    enabled: bool = True
    texture_mode: Literal["solid_random", "procedural", "image_folder", "mixed", "shadow_catcher", "palette"] = "mixed"
    palette: List[List[float]] = Field(default_factory=list)   # table colours (e.g. measured from real photos)
    saturation_max: float = 0.7           # random / procedural ground colours (real tables are rarely colourful)
    catcher_strength: Range = Field(default_factory=lambda: R(0.5, 0.85))  # shadow_catcher: shadow darkness
    texture_dir: str = ""                 # images or rectified photo textures (+ .json sidecar for size)
    texture_size_m: Range = Field(default_factory=lambda: R(0.15, 0.6))  # physical size of one texture tile (if no sidecar)
    extent_m: float = 6.0                 # half size of plane
    roughness: Range = Field(default_factory=lambda: R(0.4, 0.95))
    rotation_deg: Range = Field(default_factory=lambda: R(0, 360))
    visible_probability: float = 1.0      # probability that the plane is rendered (it always supports physics)


class BackgroundConfig(_Base):
    mode: Literal["solid_random", "gradient", "noise", "image_folder", "transparent", "mixed"] = "mixed"
    saturation_max: float = 1.0           # random backdrop colours
    image_dir: str = ""
    random_crop: bool = True
    photos_are_camera_images: bool = True  # image-folder photos keep their real look (composited after the sensor model)
    crop_scale: Range = Field(default_factory=lambda: R(0.4, 1.0))  # crop size relative to the photo (1 = whole photo)
    flip: bool = True
    brightness: Range = Field(default_factory=lambda: R(0.6, 1.2))


class LightingConfig(_Base):
    num_lights: IntRange = Field(default_factory=lambda: IntRange(min=1, max=3))
    light_type: Literal["point", "directional", "mixed"] = "mixed"
    intensity: Range = Field(default_factory=lambda: R(2.0, 8.0))          # point: W-ish units at 1 m; directional: irradiance
    color_temperature: Range = Field(default_factory=lambda: R(3500, 7500))  # Kelvin
    distance: Range = Field(default_factory=lambda: R(0.5, 1.5))
    elevation_deg: Range = Field(default_factory=lambda: R(25, 85))
    ambient: Range = Field(default_factory=lambda: R(0.1, 0.5))
    shadows: bool = True
    shadow_map_size: int = 2048
    shadow_softness: Range = Field(default_factory=lambda: R(1.0, 3.0))  # penumbra radius (shadow-map texels); large = diffuse/area light
    relight_per_view: bool = False


class SceneConfig(_Base):
    ground: GroundConfig = Field(default_factory=GroundConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)
    lighting: LightingConfig = Field(default_factory=LightingConfig)


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
class DepthConfig(_Base):
    enabled: bool = True
    model: Literal["ideal", "stereo", "tof"] = "stereo"
    baseline_m: float = 0.05
    disparity_noise_px: float = 0.08      # stereo subpixel RMS
    tof_noise_rel: float = 0.0015          # ToF sigma = rel*z + abs
    tof_noise_abs: float = 0.001
    min_z: float = 0.1
    max_z: float = 6.0
    edge_dropout: float = 0.5             # probability that a depth-edge pixel is invalid
    edge_threshold: float = 0.02          # metres jump considered an edge
    random_dropout: float = 0.002         # fraction of random holes


class CameraConfig(_Base):
    name: str = "cam0"
    preset: str = "custom"
    width: int = 1280
    height: int = 720
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 639.5
    cy: float = 359.5
    # OpenCV distortion: k1 k2 p1 p2 k3 [k4 k5 k6]
    dist: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    intrinsics_jitter: float = 0.0        # relative random perturbation of fx,fy,cx,cy per view
    # pose of this camera in the rig frame (rig frame == OpenCV frame of the sampled viewpoint)
    rig_translation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rig_rotation_deg: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # xyz euler
    # fixed world pose (T_world_cam, 4x4 row-major) used by viewpoint mode "fixed"
    world_pose: Optional[List[List[float]]] = None
    depth: DepthConfig = Field(default_factory=DepthConfig)
    calibration_file: str = ""

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def has_distortion(self) -> bool:
        return any(abs(d) > 1e-12 for d in self.dist)


def default_camera() -> CameraConfig:
    from .cameras import camera_from_preset  # lazy: cameras imports this module
    return camera_from_preset("Intel RealSense D435 / D435i", "color", 1280, 720, name="d435")


class ViewpointConfig(_Base):
    mode: Literal["sphere_random", "sphere_fibonacci", "grid", "orbit", "fixed"] = "sphere_fibonacci"
    views_per_scene: int = 8
    rig_mode: Literal["rigid", "independent"] = "rigid"
    distance_mode: Literal["absolute", "fill_fraction"] = "absolute"
    radius: Range = Field(default_factory=lambda: R(0.30, 0.60))         # absolute camera distance (m); D435 min-Z ~0.28 m
    fill_fraction: Range = Field(default_factory=lambda: R(0.6, 1.2))    # scene bounding sphere / half FOV
    elevation_deg: Range = Field(default_factory=lambda: R(25, 90))
    azimuth_deg: Range = Field(default_factory=lambda: R(-180, 180))
    roll_deg: Range = Field(default_factory=lambda: R(-20, 20))
    look_at_jitter: float = 0.015         # metres (std) added to target point
    target: Literal["scene_center", "origin", "random_object"] = "random_object"
    grid_azimuth_steps: int = 8
    grid_elevation_steps: int = 3
    # fixed mode jitter
    fixed_jitter_translation: float = 0.0  # metres std
    fixed_jitter_rotation_deg: float = 0.0
    min_visible_objects: int = 1          # reject views seeing fewer annotated objects (projected centre in image)
    max_view_attempts: int = 20


class RenderConfig(_Base):
    msaa_samples: int = 4
    near: float = 0.01
    far: float = 20.0
    canvas_padding: float = 0.15          # extra render border (fraction) -> amodal boxes / truncation
    smooth_angle_deg: float = 30.0        # crease angle for shading CAD meshes
    gpu_postprocess: bool = True          # sensor model (blur/noise/tonemap/distortion) as GLSL passes


class SensorConfig(_Base):
    enabled: bool = True
    auto_exposure: bool = True            # like a real camera: expose every image to a target brightness
    ae_target: Range = Field(default_factory=lambda: R(0.38, 0.55))  # mean brightness of the final image (0..1)
    # "light": meter the scene lighting (an 18 % grey card at the scene centre is shown at ae_target), so the parts
    #          keep the same brightness whatever the table/background looks like (a dark table just looks dark)
    # "frame": meter the whole image (mean brightness = ae_target); a bright table makes the parts dark and vice versa
    ae_metering: Literal["light", "frame"] = "light"
    exposure: Range = Field(default_factory=lambda: R(0.7, 1.4, "loguniform"))  # used when auto exposure is off
                                                                                # (and for parts on real photos)
    white_balance_jitter: float = 0.06    # +- per channel gain
    white_balance_bias: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])  # mean colour cast (R, G, B gains)
    vignetting: Range = Field(default_factory=lambda: R(0.0, 0.3))
    tonemap: Literal["srgb", "filmic"] = "filmic"
    defocus_prob: float = 0.2
    defocus_sigma: Range = Field(default_factory=lambda: R(0.3, 1.5))
    motion_blur_prob: float = 0.1
    motion_blur_length: Range = Field(default_factory=lambda: R(3, 9))
    gaussian_noise: Range = Field(default_factory=lambda: R(0.0, 0.015))
    shot_noise: Range = Field(default_factory=lambda: R(0.0, 0.01))
    saturation: Range = Field(default_factory=lambda: R(0.85, 1.15))
    contrast: Range = Field(default_factory=lambda: R(0.9, 1.1))
    jpeg_quality: Range = Field(default_factory=lambda: R(100, 100))  # <100 adds compression artefacts


class OutputConfig(_Base):
    output_dir: str = "output/dataset"
    formats: List[str] = Field(default_factory=lambda: ["native", "bop"])
    # available: see ALL_FORMATS
    rgb_format: Literal["png", "jpg"] = "jpg"
    jpg_quality: int = 95
    save_depth: bool = False             # 16-bit depth PNGs are large (~0.6 MB each); enable for RGB-D methods
    depth_scale_mm: float = 0.1           # stored uint16 = depth_mm / depth_scale_mm (BOP convention)
    save_instance_masks: bool = True
    save_semantic_masks: bool = False
    save_normals: bool = False
    save_amodal_masks: bool = False       # per-object full silhouettes (always written for BOP)
    min_visib_fract: float = 0.1          # annotations below are flagged "ignore" in detector exports
    min_px_visib: int = 50
    max_truncation: float = 0.6           # fraction of object outside image
    obb_source: Literal["visible_mask", "amodal_mask", "projected_3d_box"] = "visible_mask"
    keypoints: Literal["bbox3d_9", "bbox3d_8"] = "bbox3d_9"
    polygon_tolerance_px: float = 1.0
    split_ratios: List[float] = Field(default_factory=lambda: [0.8, 0.2, 0.0])  # train/val/test, by scene


class GenerationConfig(_Base):
    num_scenes: int = 250                 # x 8 views = 2000 images per camera
    seed: int = 42
    workers: int = 0                      # 0 = auto (all CPU cores but one; physics/encoding are CPU-bound)
    start_scene: int = 0


class ProjectConfig(_Base):
    name: str = "synthetic_dataset"
    look_profile: str = ""                # camera look profile last applied (looks/*.json), informational
    objects: ObjectsConfig = Field(default_factory=ObjectsConfig)
    placement: PlacementConfig = Field(default_factory=PlacementConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    cameras: List[CameraConfig] = Field(default_factory=lambda: [default_camera()])
    viewpoints: ViewpointConfig = Field(default_factory=ViewpointConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                import json
                json.dump(self.to_dict(), f, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # yaml is a superset of json
        cfg = cls.model_validate(data or {})
        cfg.resolve_paths(path.parent)
        return cfg

    def resolve_paths(self, base: Path) -> None:
        """Make relative paths in a loaded config relative to the config file."""

        def fix(p: str) -> str:
            if not p:
                return p
            if os.name != "nt" and "\\" in p and not (len(p) > 1 and p[1] == ":"):
                p = p.replace("\\", "/")      # relative path written on Windows
            pp = Path(p)
            if pp.is_absolute():
                return p
            cand = (base / pp).resolve()
            return str(cand) if cand.exists() or not pp.exists() else p

        for m in self.objects.models:
            m.path = fix(m.path)
        self.objects.distractors.mesh_dir = fix(self.objects.distractors.mesh_dir)
        self.scene.ground.texture_dir = fix(self.scene.ground.texture_dir)
        self.scene.background.image_dir = fix(self.scene.background.image_dir)
        for c in self.cameras:
            c.calibration_file = fix(c.calibration_file)
        od = Path(self.output.output_dir)
        if self.output.output_dir and not od.is_absolute():
            self.output.output_dir = str((base / od).resolve())

    def copy_deep(self) -> "ProjectConfig":
        return ProjectConfig.model_validate(self.to_dict())

    def enabled_models(self) -> List[ObjectModel]:
        return [m for m in self.objects.models if m.enabled and m.path]


ALL_FORMATS = ["native", "bop", "coco", "coco_bb", "yolo", "yolo_obb", "yolo_seg", "yolo_pose"]
# formats that are (re-)built from the native annotations after rendering; native and BOP are written per scene
DETECTOR_FORMATS = ["coco", "coco_bb", "yolo", "yolo_obb", "yolo_seg", "yolo_pose"]
