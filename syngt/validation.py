"""Pre-flight checks of a project configuration.

Every place where the generator would silently fall back to something else, or where a setting has no effect
in the current mode, is reported here in plain language - so nothing "just doesn't happen" without a message.
No Qt, no rendering: cheap enough to run after every edit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .config import ProjectConfig

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass(frozen=True)
class Issue:
    severity: str      # "error" (generation would fail / be useless), "warning" (probably not what you want), "info"
    message: str       # one plain sentence
    where: str = ""    # settings tab that fixes it ("Objects", "Scene & Light", ...)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}" + (f" ({self.where})" if self.where else "")


def _has_images(folder: str) -> bool:
    from .textures import list_images
    try:
        return bool(folder) and bool(list_images(folder))
    except Exception:
        return False


def check_config(cfg: ProjectConfig, extent_of: Optional[Callable[[str, str], Optional[float]]] = None) -> List[Issue]:
    """All problems of `cfg`, most severe first. `extent_of(path, unit)` -> largest model dimension in metres
    (optional: enables the unit sanity check; it loads the mesh, so callers pass a cached function)."""
    out: List[Issue] = []

    def add(sev, msg, where=""):
        out.append(Issue(sev, msg, where))

    # ------------------------------------------------------------------ objects
    models = [m for m in cfg.objects.models if m.enabled]
    if not cfg.enabled_models():
        add(ERROR, "No part is enabled: add at least one CAD file.", "Objects")
    for m in models:
        name = m.class_name
        if not m.path:
            add(ERROR, f"Part '{name}' has no file.", "Objects")
            continue
        if not Path(m.path).exists():
            add(ERROR, f"Part '{name}': file not found: {m.path}", "Objects")
            continue
        if m.material.mode == "palette" and not m.material.palette:
            add(WARNING, f"Part '{name}': colour mode is 'palette' but the palette is empty - its single colour "
                         f"is used instead.", "Objects")
        if extent_of is not None:
            ext = extent_of(m.path, m.unit)
            if ext is not None and (ext < 0.002 or ext > 2.0):
                add(WARNING, f"Part '{name}' would be {ext * 1000:.4g} mm large with file unit '{m.unit}' - "
                             f"check the unit.", "Objects")
        if m.count.max < m.count.min:
            add(WARNING, f"Part '{name}': instances max ({m.count.max}) is below min ({m.count.min}); "
                         f"the two are swapped.", "Objects")
    # class names: what a training script will see
    by_id, by_name = {}, {}
    for m in models:
        by_id.setdefault(m.class_id, []).append(m.class_name)
        by_name.setdefault(m.class_name, set()).add(m.class_id)
    for name, ids in by_name.items():
        if len(ids) > 1:
            add(WARNING, f"Class name '{name}' is used for {len(ids)} different class IDs {sorted(ids)} - the "
                         f"classes cannot be told apart by name. Give the parts different names or the same ID.",
                "Objects")
    for cid, names in by_id.items():
        if len(set(names)) > 1:
            add(INFO, f"Parts {', '.join(sorted(set(names)))} share class ID {cid}: all of them are labelled "
                      f"'{names[0]}'.", "Objects")
    oc = cfg.objects
    if models and oc.count_mode == "per_model" and all(m.count.max <= 0 and m.count.min <= 0 for m in models):
        add(WARNING, "Every part has 0 instances per scene - the images will contain no labelled part.", "Objects")
    if models and oc.count_mode == "total" and sum(max(m.weight, 0) for m in models) <= 0:
        add(WARNING, "Count mode 'total' with all weights 0 - parts are chosen with equal probability.", "Objects")
    d = oc.distractors
    if d.enabled and d.count.max > 0 and not d.shapes and not d.mesh_dir:
        add(WARNING, "Distractors are enabled but no shape and no mesh folder is selected - none are created.",
            "Objects")
    if d.enabled and d.material.mode == "palette" and not d.material.palette:
        add(WARNING, "Distractor colour mode is 'palette' but the palette is empty.", "Objects")

    # ------------------------------------------------------------------ scene
    g, b, lg = cfg.scene.ground, cfg.scene.background, cfg.scene.lighting
    floating = cfg.placement.mode == "free_floating"
    if g.enabled:
        if g.texture_mode == "image_folder" and not _has_images(g.texture_dir):
            add(WARNING, "Table texture mode is 'image folder' but the folder has no images - procedural "
                         "textures are used instead.", "Scene & Light")
        if g.texture_mode == "palette" and not g.palette:
            add(WARNING, "Table mode is 'table colours' but no colours are set - procedural textures are used "
                         "instead.", "Scene & Light")
        if g.texture_mode == "shadow_catcher" and not (b.mode in ("image_folder", "mixed") and _has_images(b.image_dir)):
            add(WARNING, "The table is an invisible shadow catcher, but there are no background photos - parts "
                         "will lie on a random backdrop.", "Scene & Light")
    elif not floating:
        add(WARNING, "The table (ground plane) is switched off but parts are placed on it - they will appear to "
                     "float in front of the backdrop.", "Scene & Light")
    if b.mode == "image_folder" and not _has_images(b.image_dir):
        add(WARNING, "Background is 'photos' but the photo folder has no images - procedural noise is used "
                     "instead.", "Scene & Light")
    if lg.num_lights.max <= 0 and lg.ambient.max <= 0:
        add(ERROR, "No lights and no ambient light: every image would be black.", "Scene & Light")

    # ------------------------------------------------------------------ cameras / views
    if not cfg.cameras:
        add(ERROR, "No camera is configured.", "Cameras")
    from .writers import safe_name
    names = [safe_name(c.name) for c in cfg.cameras]       # the form used in file names
    for n in sorted({n for n in names if names.count(n) > 1}):
        add(ERROR, f"Two cameras have the same file name '{n}' - their images would overwrite each other. "
                   f"Rename one.", "Cameras")
    v = cfg.viewpoints
    if v.mode == "fixed":
        for c in cfg.cameras:
            if c.world_pose is None:
                add(WARNING, f"Viewpoint mode is 'fixed' but camera '{c.name}' has no world pose - a default pose "
                             f"(about 0.7 m away, 45 deg) is used.", "Cameras")
    elif v.distance_mode == "absolute":
        for c in cfg.cameras:
            if c.depth.enabled and cfg.output.save_depth and v.radius.min < c.depth.min_z:
                add(WARNING, f"Camera '{c.name}' measures no depth closer than {c.depth.min_z * 100:.0f} cm, but "
                             f"views start at {v.radius.min * 100:.0f} cm - near views get empty depth.",
                    "Viewpoints")
    if v.mode == "grid" and v.grid_azimuth_steps * v.grid_elevation_steps != v.views_per_scene:
        add(INFO, f"Viewpoint mode 'grid' renders {v.grid_azimuth_steps} x {v.grid_elevation_steps} = "
                  f"{v.grid_azimuth_steps * v.grid_elevation_steps} views per scene ('views per scene' is not used).",
            "Viewpoints")

    # ------------------------------------------------------------------ sensor / output
    s, o = cfg.sensor, cfg.output
    if not s.enabled and cfg.look_profile:
        add(WARNING, "A camera look profile is selected but sensor effects are switched off - the look is not "
                     "applied.", "Sensor")
    if "bop" in o.formats:
        # BOP stores a rigid pose + one model per object + a pinhole camera: it cannot express these
        jit = [m.class_name for m in models
               if abs(m.scale_jitter.min - 1) > 1e-6 or abs(m.scale_jitter.max - 1) > 1e-6]
        if jit:
            add(WARNING, f"BOP export with scale jitter ({', '.join(jit[:3])}{' …' if len(jit) > 3 else ''}): BOP "
                         f"poses carry no scale, so they will not match the images. Set scale jitter to 1 or "
                         f"drop BOP.", "Objects")
        for c in cfg.cameras:
            if c.has_distortion():
                add(WARNING, f"BOP export with lens distortion (camera '{c.name}'): BOP files only hold a pinhole "
                             f"camera, so BOP poses will not project onto the distorted images. Use the native "
                             f"annotations (they include the distortion) or set the distortion to 0.", "Cameras")
    if o.save_depth:
        for c in cfg.cameras:
            if c.depth.enabled and c.depth.max_z * 1000.0 / max(o.depth_scale_mm, 1e-9) > 65535:
                add(WARNING, f"Camera '{c.name}': depths beyond {65535 * o.depth_scale_mm / 1000:.2f} m do not fit "
                             f"into the 16-bit depth image at {o.depth_scale_mm} mm per unit and are stored as "
                             f"'no measurement' (raise the depth scale to keep them).", "Output")
    if o.save_depth and cfg.cameras and not any(c.depth.enabled for c in cfg.cameras):
        add(WARNING, "'Save depth' is on but depth rendering is off for every camera - no depth is written.",
            "Output")
    if sum(max(r, 0) for r in (o.split_ratios + [0, 0, 0])[:3]) <= 0:
        add(WARNING, "All split ratios are 0 - every scene goes to 'train'.", "Output")
    from .config import ALL_FORMATS
    unknown = [f for f in o.formats if f not in ALL_FORMATS]
    if "coco_bb" in o.formats and (o.split_ratios + [0, 0])[1] <= 0:
        add(WARNING, "The RT-DETR / COCO-boxes export needs a validation set, but the validation split is 0.",
            "Output")
    if unknown:
        add(WARNING, f"Unknown output format(s) {unknown} are ignored.", "Output")
    if cfg.generation.num_scenes <= 0:
        add(ERROR, "Number of scenes is 0.", "Output")

    order = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(out, key=lambda i: order[i.severity])
