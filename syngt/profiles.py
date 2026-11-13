"""User-friendly presets for camera viewpoints and object layout.

Each profile only touches the settings it is about; everything else stays as configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .config import ProjectConfig, R


@dataclass
class Profile:
    title: str
    description: str
    apply: Callable[[ProjectConfig], None]


# --------------------------------------------------------------------------- viewpoints
def _view(mode="sphere_fibonacci", dist=(0.30, 0.60), elev=(25, 90), roll=(-20, 20), az=(-180, 180),
          target="random_object", jitter=0.015, views=8):
    def f(cfg: ProjectConfig):
        v = cfg.viewpoints
        v.mode = mode
        v.distance_mode = "absolute"
        v.radius = R(*dist)
        v.elevation_deg = R(*elev)
        v.azimuth_deg = R(*az)
        v.roll_deg = R(*roll)
        v.target = target
        v.look_at_jitter = jitter
        v.views_per_scene = views
    return f


def _fixed(cfg: ProjectConfig):
    v = cfg.viewpoints
    v.mode = "fixed"
    v.fixed_jitter_translation = 0.01
    v.fixed_jitter_rotation_deg = 2.0
    v.views_per_scene = 4
    from .geometry import look_at_cv
    for i, c in enumerate(cfg.cameras):
        if c.world_pose is None:  # sensible default: 45 deg, 50 cm, looking at the table centre
            import math
            a = math.radians(-90 + 30 * i)
            c.world_pose = look_at_cv([0.35 * math.cos(a), 0.35 * math.sin(a), 0.35], [0, 0, 0]).tolist()


VIEW_PROFILES: Dict[str, Profile] = {
    "full": Profile("Full variety (recommended start)",
                    "Cameras all around and above the parts, 30-60 cm away. Maximum viewpoint diversity for "
                    "a pose estimator that must work from anywhere.",
                    _view(dist=(0.30, 0.60), elev=(25, 90), roll=(-20, 20))),
    "top": Profile("Top view (more or less)",
                   "Camera looks down on the table (70-90 deg), any in-plane rotation. Typical fixed overhead "
                   "camera, conveyor or bin-picking cell.",
                   _view(dist=(0.40, 0.70), elev=(70, 90), roll=(-180, 180), target="scene_center", jitter=0.03)),
    "overview45": Profile("45 deg overview",
                          "Oblique view from about 45 deg (35-55 deg), all around the scene, 40-80 cm away. "
                          "Like a camera on a tripod next to the workspace.",
                          _view(dist=(0.40, 0.80), elev=(35, 55), roll=(-10, 10), target="scene_center")),
    "robot": Profile("Robot-mounted (wrist / eye-in-hand)",
                     "Close-up views from 15-35 cm, steep to oblique angles (40-90 deg), arbitrary camera roll, "
                     "aimed at individual parts. Note: D435 depth is invalid closer than ~28 cm - use the "
                     "D405 preset for close-range depth.",
                     _view(dist=(0.15, 0.35), elev=(40, 90), roll=(-180, 180), target="random_object", jitter=0.02,
                           views=10)),
    "low": Profile("Low / side view",
                   "Grazing angles (10-30 deg) - strong perspective and self-occlusion, parts seen from the side.",
                   _view(dist=(0.30, 0.60), elev=(10, 30), roll=(-10, 10))),
    "fixed": Profile("Fixed camera (calibrated pose)",
                     "Every camera stays at its world pose (Cameras tab -> World pose, or imported calibration), "
                     "with small jitter. Use for a known, fixed installation.",
                     _fixed),
    "turntable": Profile("Turntable / orbit",
                         "Camera circles the scene at a fixed height and distance per scene (12 steps) - "
                         "sequence-like data, even azimuth coverage.",
                         _view(mode="orbit", dist=(0.35, 0.55), elev=(25, 60), roll=(0, 0), target="scene_center",
                               jitter=0.0, views=12)),
}


# --------------------------------------------------------------------------- layout
def _layout(mode, cluster=None, area=None, orient="random", seq=False, settle=True, spacing=0.005):
    def f(cfg: ProjectConfig):
        p = cfg.placement
        p.mode = mode
        if cluster is not None:
            p.cluster_radius = R(*cluster)
        if area is not None:
            p.area_size = list(area)
        p.initial_orientation = orient
        p.sequential_drop = seq
        p.settle_with_physics = settle
        p.min_spacing = spacing
        cfg.scene.ground.enabled = mode != "free_floating"
    return f


LAYOUT_PROFILES: Dict[str, Profile] = {
    "scattered": Profile("Scattered on a table",
                         "Parts dropped over the table, mostly lying apart in natural resting poses, some touching.",
                         _layout("physics_drop", cluster=(0.05, 0.10), area=(0.25, 0.20))),
    "pile": Profile("Pile (touching / leaning / occluded)",
                    "Parts dropped close together: they lean on each other and occlude - hard cases for pose.",
                    _layout("physics_drop", cluster=(0.005, 0.03), area=(0.25, 0.20), seq=True)),
    "separated": Profile("Separated, stable poses",
                         "Every part in one of its stable resting poses, never touching (min 5 mm gap). Clean data.",
                         _layout("stable_separated", area=(0.25, 0.20))),
    "bin": Profile("Bin picking",
                   "Parts dropped into a box with walls (Placement tab -> bin size).",
                   _layout("physics_bin", cluster=(0.02, 0.06))),
    "floating": Profile("Floating, any orientation",
                        "No table: parts float with fully random 3D orientation. Covers every view of a part "
                        "(also from below); combine with 'Full variety' viewpoints.",
                        _layout("free_floating")),
}


def apply_view_profile(cfg: ProjectConfig, key: str) -> None:
    VIEW_PROFILES[key].apply(cfg)


def apply_layout_profile(cfg: ProjectConfig, key: str) -> None:
    LAYOUT_PROFILES[key].apply(cfg)
    if key == "floating":
        v = cfg.viewpoints
        v.elevation_deg = R(-80, 90)


# --------------------------------------------------------------------------- look (appearance)
def _look(ground_mode, ground_sat, bg_mode, bg_sat, n_lights, light_type, elev, intensity, ambient, softness,
          defocus_prob, defocus_sigma, exposure, noise, saturation, wb, distractor_sat):
    def f(cfg: ProjectConfig):
        g, b, l, s = cfg.scene.ground, cfg.scene.background, cfg.scene.lighting, cfg.sensor
        g.texture_mode, g.saturation_max = ground_mode, ground_sat
        b.mode, b.saturation_max = bg_mode, bg_sat
        # photos of the real cell come from the same camera: crop only slightly (keeps the true scale)
        b.crop_scale = R(0.85, 1.0) if ground_mode == "shadow_catcher" else R(0.4, 1.0)
        if bg_mode == "image_folder" and not b.image_dir:
            b.mode = "mixed"  # no photos yet -> fall back
            if ground_mode == "shadow_catcher":
                g.texture_mode = "mixed"
        l.num_lights.min, l.num_lights.max = n_lights
        l.light_type = light_type
        l.elevation_deg, l.intensity, l.ambient, l.shadow_softness = R(*elev), R(*intensity), R(*ambient), R(*softness)
        s.enabled = True
        s.defocus_prob, s.defocus_sigma = defocus_prob, R(*defocus_sigma)
        s.exposure = R(*exposure, "loguniform")
        s.gaussian_noise, s.shot_noise = R(*noise), R(0.0, noise[1])
        s.saturation, s.white_balance_jitter = R(*saturation), wb
        cfg.objects.distractors.material.random_saturation_max = distractor_sat
    return f


LOOK_PROFILES: Dict[str, Profile] = {
    "realistic": Profile("Realistic workcell (your photos as background)",
                         "Objects rendered onto photos of your real, empty workspace (Background folder - e.g. clean "
                         "plates built from your annotated images) with an invisible shadow-catching table, soft "
                         "ceiling-like light, real camera softness. Best with a top view matching the photos.",
                         _look("shadow_catcher", 0.15, "image_folder", 1.0, (1, 2), "directional", (55, 90),
                               (1.5, 4.0), (0.5, 1.0), (4, 12), 0.7, (0.5, 1.5), (0.55, 1.0), (0.001, 0.006),
                               (0.8, 1.05), 0.05, 0.3)),
    "neutral": Profile("Neutral indoor (white / grey / black tables)",
                       "Plain, low-saturation tables and backdrops like a lab bench, soft shadows, mild camera "
                       "softness. Close to typical industrial images without needing photos.",
                       _look("mixed", 0.12, "mixed", 0.25, (1, 3), "mixed", (40, 90), (1.5, 5.0), (0.35, 0.9),
                             (3, 10), 0.5, (0.4, 1.3), (0.6, 1.1), (0.001, 0.008), (0.8, 1.1), 0.06, 0.35)),
    "random": Profile("Domain randomisation (varied colours)",
                      "Colourful random tables, textures and backdrops, hard and soft shadows. Maximum appearance "
                      "variety - robust, but far from any specific real setup.",
                      _look("mixed", 0.7, "mixed", 1.0, (1, 3), "mixed", (25, 85), (2.0, 8.0), (0.1, 0.5), (1, 6),
                            0.2, (0.3, 1.5), (0.7, 1.4), (0.0, 0.015), (0.85, 1.15), 0.06, 1.0)),
}


def apply_look_profile(cfg: ProjectConfig, key: str) -> None:
    LOOK_PROFILES[key].apply(cfg)


# --------------------------------------------------------------------------- background (separate from the camera)
def _bg(ground_mode, ground_sat, bg_mode, bg_sat, crop=(0.4, 1.0)):
    def f(cfg: ProjectConfig):
        g, b = cfg.scene.ground, cfg.scene.background
        if cfg.placement.mode != "free_floating":   # floating parts have no table: a background choice keeps it off
            g.enabled = True
        g.texture_mode, g.saturation_max = ground_mode, ground_sat
        b.mode, b.saturation_max = bg_mode, bg_sat
        b.crop_scale = R(*crop)
    return f


BACKGROUND_PROFILES: Dict[str, Profile] = {
    "photos": Profile("Real-world background (photos)",
                      "Parts are rendered onto photos of your real, empty workspace; an invisible table catches "
                      "their shadows. Choose the photo folder below (or build clean plates from annotated photos). "
                      "Fits viewpoints similar to the photos (e.g. top view).",
                      _bg("shadow_catcher", 0.15, "image_folder", 1.0, (0.85, 1.0))),
    "table_colours": Profile("Real table colours (measured)",
                             "Synthetic table in the colours measured from your real photos (with subtle texture). "
                             "Works for every viewpoint; no empty-workspace photos needed.",
                             _bg("palette", 0.2, "mixed", 0.25)),
    "neutral": Profile("Neutral tables (white / grey / black)",
                       "Plain, low-saturation synthetic tables and backdrops like a typical lab bench.",
                       _bg("mixed", 0.12, "mixed", 0.25)),
    "random": Profile("Random colourful (domain randomisation)",
                      "Colourful random tables, textures and backdrops - maximum variety, not tied to a real setup.",
                      _bg("mixed", 0.7, "mixed", 1.0)),
}


def apply_background_profile(cfg: ProjectConfig, key: str) -> None:
    BACKGROUND_PROFILES[key].apply(cfg)


def current_background_profile(cfg: ProjectConfig) -> str:
    g, b = cfg.scene.ground, cfg.scene.background
    if g.texture_mode == "shadow_catcher" or b.mode == "image_folder":
        return "photos"
    if g.texture_mode == "palette":
        return "table_colours"
    if g.saturation_max <= 0.2:
        return "neutral"
    return "random"


def randomise_camera(cfg: ProjectConfig) -> None:
    """Camera colour & lighting: wide random ranges instead of a measured profile (domain randomisation)."""
    l, s = cfg.scene.lighting, cfg.sensor
    l.num_lights.min, l.num_lights.max = 1, 3
    l.light_type = "mixed"
    # varied but camera-friendly: direct light at most ~3x the ambient level (otherwise no exposure can show
    # both white plastic and a dark table, as in real indoor scenes)
    l.elevation_deg, l.intensity, l.ambient = R(30, 85), R(0.8, 3.0), R(0.4, 1.0)
    l.shadow_softness, l.color_temperature = R(2, 12), R(3500, 7000)
    s.enabled = True
    s.auto_exposure, s.ae_target = True, R(0.32, 0.62)   # varied, but never blown out / black
    s.exposure = R(0.3, 0.8, "loguniform")
    s.white_balance_bias, s.white_balance_jitter = [1.0, 1.0, 1.0], 0.08
    s.gaussian_noise, s.shot_noise = R(0.0, 0.015), R(0.0, 0.01)
    s.defocus_prob, s.defocus_sigma = 0.3, R(0.3, 1.5)
    s.saturation, s.contrast = R(0.8, 1.2), R(0.85, 1.15)
    cfg.look_profile = ""
