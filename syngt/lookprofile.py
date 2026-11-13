"""Camera look profiles: measured once from real photos, re-usable for any project.

A profile stores
  * the measured statistics of the real images (brightness and its spread, contrast, saturation, colour cast,
    noise, sharpness),
  * the colours of the support surfaces (k-means over pixels that are not annotated objects),
  * sensor settings that were auto-tuned in a closed loop (render -> measure -> adjust) until synthetic
    images match the real statistics.

Applying a profile writes these settings into a project. Scene content (layout, viewpoints, objects) is not
touched, so one profile per camera / workcell can be used for many datasets.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .appearance import image_stats, list_images
from .config import R, ProjectConfig

LOOKS_DIR = Path(__file__).resolve().parents[1] / "looks"
METRICS = ["brightness", "contrast", "saturation", "cast_r", "cast_b", "noise", "sharpness"]
SENSOR_KEYS = ["exposure", "auto_exposure", "ae_metering", "ae_target", "white_balance_bias", "white_balance_jitter", "gaussian_noise", "shot_noise",
               "defocus_prob", "defocus_sigma", "saturation", "contrast", "vignetting", "tonemap"]


def _summ(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        out[k] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    return out


# --------------------------------------------------------------------------- real images
def _folders(folder) -> List[str]:
    return [folder] if isinstance(folder, (str, Path)) else list(folder)


def measure_real(folder, limit: int = 150, log: Callable[[str], None] = print):
    """Statistics + surface colours of real images (one folder or a list of folders).
    Annotated objects are masked if labels exist (YOLO .txt / COCO json)."""
    from .plates import _coco_polys, object_mask
    folders = _folders(folder)
    files, coco = [], {}
    for f in folders:
        files += list_images(f, 10**6)
        coco.update(_coco_polys(f))
    if len(files) > limit:
        files = files[:: max(1, len(files) // limit)][:limit]
    if not files:
        raise RuntimeError(f"no images in {folders}")
    rows, pix = [], []
    for f in files:
        im = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rows.append(image_stats(rgb))
        mask = object_mask(f, rgb.shape, coco, 15)
        small = cv2.resize(rgb, (160, 100), interpolation=cv2.INTER_AREA)
        ms = cv2.resize(mask.astype(np.uint8), (160, 100), interpolation=cv2.INTER_NEAREST).astype(bool)
        pix.append(small[~ms].reshape(-1, 3))
    log(f"measured {len(rows)} real images")
    return _summ(rows), surface_palette(np.concatenate(pix))


def surface_palette(pixels: np.ndarray, k: int = 6, min_weight: float = 0.08) -> List[Dict]:
    """Dominant surface colours (sRGB 0..1) with their area share."""
    data = pixels.astype(np.float32)
    if len(data) > 60000:
        data = data[np.random.default_rng(0).choice(len(data), 60000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    w = np.bincount(labels.ravel(), minlength=k) / len(labels)
    out = [{"color": [float(x) / 255 for x in c], "weight": float(wi)} for c, wi in zip(centers, w) if wi >= min_weight]
    return sorted(out, key=lambda d: -d["weight"])


# --------------------------------------------------------------------------- synthetic
def measure_synth(cfg: ProjectConfig, n_frames: int = 10, seed0: int = 7000, sampler=None):
    """Render n frames with fixed seeds (so successive iterations compare the same scenes)."""
    from .scene import SceneSampler
    s = sampler or SceneSampler(cfg)
    s.cfg = cfg
    rows = []
    for k in range(n_frames):
        st = s.sample_scene(seed0 + k, np.random.default_rng([seed0, k]))
        fr = s.render_view(st, 0, np.random.default_rng([seed0, k, 1]))[0]
        rows.append(image_stats(fr.rgb[..., :3]))
    return _summ(rows), s


# --------------------------------------------------------------------------- closed-loop matching
def _rel_err(real, synth, k):
    a, b = real[k]["mean"], synth[k]["mean"]
    if k in ("cast_r", "cast_b"):
        return abs(a - b)
    if k == "noise":
        return abs(a - b) / max(a, 0.002)
    return abs(a - b) / max(abs(a), 1e-6)


def _score(errs: Dict[str, float]) -> float:
    w = {"brightness": 4.0, "contrast": 1.5, "saturation": 1.0, "cast_r": 8.0, "cast_b": 8.0, "noise": 0.5,
         "sharpness": 0.7}
    return float(sum(w[k] * errs[k] for k in w))


def tune_sensor(cfg: ProjectConfig, real: Dict, iterations: int = 7, n_frames: int = 10,
                log: Callable[[str], None] = print, progress: Callable[[int, int], None] = lambda i, n: None):
    """Adjust cfg.sensor until synthetic statistics match `real`. Returns (synth stats, history).
    Exposure is tuned with auto exposure off (that exposure is used for parts on real photos); afterwards auto
    exposure is switched on with the real brightness range as target, so any lighting is exposed correctly."""
    s = cfg.sensor
    s.enabled = True
    s.auto_exposure = False
    # spread of real brightness -> width of the exposure range (in linear space)
    spread = (real["brightness"]["p90"] / max(real["brightness"]["p10"], 1e-3)) ** 2.2
    spread = float(np.clip(spread, 1.15, 2.5))
    s.white_balance_jitter = float(np.clip(max(abs(real["cast_r"]["p90"] - real["cast_r"]["p10"]),
                                               abs(real["cast_b"]["p90"] - real["cast_b"]["p10"])) / 2, 0.01, 0.1))
    if s.gaussian_noise.max <= 0:
        s.gaussian_noise = R(0.0, 0.004)
    sampler, hist = None, []
    synth = None
    best = None                    # (score, sensor snapshot, synth stats)
    exp_hist = []                  # (log exposure centre, brightness) for the secant update
    for it in range(iterations):
        progress(it, iterations)
        synth, sampler = measure_synth(cfg, n_frames, sampler=sampler)
        errs = {k: _rel_err(real, synth, k) for k in METRICS}
        hist.append({"iteration": it, "synth": {k: synth[k]["mean"] for k in METRICS}, "errors": errs})
        log(f"iteration {it}: " + ", ".join(f"{k} {synth[k]['mean']:.3g}/{real[k]['mean']:.3g}" for k in METRICS))
        sc = _score(errs)
        if best is None or sc < best[0]:
            best = (sc, s.model_copy(deep=True), synth)
        ok = (errs["brightness"] < 0.04 and errs["contrast"] < 0.08 and errs["saturation"] < 0.15
              and errs["cast_r"] < 0.02 and errs["cast_b"] < 0.02 and errs["sharpness"] < 0.25)
        if ok:
            log("converged")
            break
        # exposure: brightness ~ exposure^(1/2.2) around mid-grey
        logc = math.log(math.sqrt(s.exposure.min * s.exposure.max))
        exp_hist.append((logc, synth["brightness"]["mean"]))
        target = real["brightness"]["mean"]
        slope = None
        if len(exp_hist) >= 2:   # secant: measured d(brightness)/d(log exposure)
            (l0, b0), (l1, b1) = exp_hist[-2], exp_hist[-1]
            if abs(l1 - l0) > 1e-3 and (b1 - b0) / (l1 - l0) > 0.01:
                slope = (b1 - b0) / (l1 - l0)
        if slope is None:        # first step: gamma model, damped
            slope = synth["brightness"]["mean"] / 2.2 * 1.6
        step = float(np.clip((target - synth["brightness"]["mean"]) / slope, -0.8, 0.8))
        c = math.exp(logc + step)
        s.exposure = R(c / math.sqrt(spread), c * math.sqrt(spread), "loguniform")
        # colour cast -> white balance bias (R/G and B/G ratios)
        b = list(s.white_balance_bias)
        b[0] *= real["cast_r"]["mean"] / max(synth["cast_r"]["mean"], 1e-3)
        b[2] *= real["cast_b"]["mean"] / max(synth["cast_b"]["mean"], 1e-3)
        s.white_balance_bias = [float(np.clip(b[0], 0.6, 1.6)), 1.0, float(np.clip(b[2], 0.6, 1.6))]
        # saturation & contrast (display-space multipliers)
        # plausible camera range only: scene colour comes from the content (tables, parts), not from the ISP
        for key, attr, lo, hi in (("saturation", "saturation", 0.7, 1.25), ("contrast", "contrast", 0.75, 1.3)):
            ratio = real[key]["mean"] / max(synth[key]["mean"], 1e-4)
            ratio = float(np.clip(ratio, 0.5, 2.0)) ** (0.6 if key == "saturation" else 0.6)
            r = getattr(s, attr)
            mid, half = 0.5 * (r.min + r.max) * ratio, max(0.5 * (r.max - r.min), 0.05)  # keep some variation
            mid = float(np.clip(mid, lo, hi))
            setattr(s, attr, R(max(lo, mid - half), min(hi, mid + half)))
        # sharpness: more / less defocus
        shr = real["sharpness"]["mean"] / max(synth["sharpness"]["mean"], 1e-3)
        if shr < 0.8:
            s.defocus_prob = float(np.clip(s.defocus_prob + 0.25, 0.0, 1.0))
            s.defocus_sigma = R(max(0.3, s.defocus_sigma.min), float(np.clip(s.defocus_sigma.max / math.sqrt(shr), 0.6, 4.0)))
        elif shr > 1.25:
            s.defocus_prob = float(np.clip(s.defocus_prob - 0.25, 0.0, 1.0))
            s.defocus_sigma = R(0.3, float(max(0.5, s.defocus_sigma.max / math.sqrt(shr))))
        # noise (measured as high-frequency residual)
        nr = real["noise"]["mean"] / max(synth["noise"]["mean"], 1e-5)
        g = s.gaussian_noise
        s.gaussian_noise = R(g.min * nr ** 0.7, float(np.clip(max(g.max, 0.001) * nr ** 0.7, 0.0005, 0.05)))
    progress(iterations, iterations)
    if best is not None:           # keep the best iteration, not the last one
        cfg.sensor = best[1]
        synth = best[2]
        log(f"best match: iteration {[h['iteration'] for h in hist][int(np.argmin([_score(h['errors']) for h in hist]))]}")
    # Auto exposure that reproduces the tuned exposure: meter the scene light (an 18 % grey card), with the
    # target = how bright that card is at the tuned exposure under the tuning lighting. Parts then keep the
    # brightness that was matched to the real photos, whatever table / background is used later.
    cfg.sensor.auto_exposure = True
    cfg.sensor.ae_metering = "light"
    cfg.sensor.ae_target = light_metering_target(cfg.sensor.exposure, cfg.scene.lighting, cfg.sensor.tonemap)
    log(f"auto exposure: grey-card target {cfg.sensor.ae_target.min:.2f}-{cfg.sensor.ae_target.max:.2f} "
        f"(equals the tuned exposure {cfg.sensor.exposure.min:.2f}-{cfg.sensor.exposure.max:.2f} for this lighting)")
    return synth, hist


def light_metering_target(exposure, lighting, tonemap: str = "filmic", n: int = 300):
    """Grey-card auto-exposure target (Range) that gives the same exposure as the fixed `exposure` range under
    `lighting` (LightingConfig). Pure calculation: lighting set-ups are sampled, nothing is rendered."""
    from .imaging import display_brightness
    from .scene import GREY_CARD, sample_lights, white_radiance
    rng = np.random.default_rng(0)
    centre = np.zeros(3)
    grey = []
    for _ in range(n):
        lights, sky, _g = sample_lights(lighting, rng, centre)
        grey.append(white_radiance(lights, sky, centre) * GREY_CARD)
    g = float(np.exp(np.mean(np.log(np.maximum(grey, 1e-6)))))       # geometric mean: exposure is multiplicative
    lo, hi = sorted((exposure.min, exposure.max))
    t = [display_brightness(np.array([[[e * g] * 3]], np.float32), tonemap) for e in (lo, hi)]
    return R(float(np.clip(t[0], 0.03, 0.95)), float(np.clip(t[1], 0.03, 0.95)))


# --------------------------------------------------------------------------- profile io
def create_profile(name: str, real_folder, cfg: ProjectConfig, iterations: int = 7, n_frames: int = 10,
                   use_surface_colors: bool = True, log: Callable[[str], None] = print,
                   progress: Callable[[int, int], None] = lambda i, n: None) -> Dict:
    real, palette = measure_real(real_folder, log=log)
    work = cfg.copy_deep()
    if use_surface_colors and not work.scene.background.image_dir:
        # without photos of the workcell the table colours come from the measured surfaces
        work.scene.ground.texture_mode = "palette"
        work.scene.ground.palette = [p["color"] for p in palette]
    synth, hist = tune_sensor(work, real, iterations, n_frames, log, progress)
    sens = work.sensor.model_dump(mode="json")
    prof = {
        "name": name, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": [str(Path(f).resolve()) for f in _folders(real_folder)],
        "n_images": sum(len(list_images(f, 10**6)) for f in _folders(real_folder)),
        "real_stats": real, "synthetic_stats": synth, "history": hist,
        "sensor": {k: sens[k] for k in SENSOR_KEYS if k in sens},
        "lighting": work.scene.lighting.model_dump(mode="json"),   # exposure was tuned for this lighting
        "tuning_scene": "measured table colours (not stored - background is a separate option)",
    }
    return prof


def save_profile(prof: Dict, path: Optional[str] = None) -> str:
    LOOKS_DIR.mkdir(exist_ok=True)
    p = Path(path) if path else LOOKS_DIR / (_slug(prof["name"]) + ".json")
    p.write_text(json.dumps(prof, indent=2), encoding="utf-8")
    return str(p)


def load_profile(path: str) -> Dict:
    prof = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prof, dict) or not isinstance(prof.get("sensor"), dict):
        raise ValueError(f"{Path(path).name} is not a camera look profile")
    return prof


def default_profile() -> Optional[str]:
    """Path of the default look profile (applied to new projects), or None."""
    marker = LOOKS_DIR / "DEFAULT"
    if marker.exists():
        name = marker.read_text(encoding="utf-8").strip()
        p = LOOKS_DIR / name
        if name and p.is_file():
            return str(p)
    return None


def set_default_profile(path: Optional[str]) -> None:
    LOOKS_DIR.mkdir(exist_ok=True)
    marker = LOOKS_DIR / "DEFAULT"
    if path:
        if not (LOOKS_DIR / Path(path).name).is_file():
            raise ValueError(f"{path} is not a profile in {LOOKS_DIR}")
        marker.write_text(Path(path).name, encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def list_profiles() -> List[str]:
    return sorted(str(p) for p in LOOKS_DIR.glob("*.json")) if LOOKS_DIR.exists() else []


def apply_profile(cfg: ProjectConfig, prof: Dict, path: str = "", lighting: bool = True) -> List[str]:
    """Write the camera look into a project: sensor settings (+ the lighting the exposure was tuned with).
    Backgrounds / table colours are NOT part of a camera look (see apply_surface_colors)."""
    from .config import LightingConfig, SensorConfig
    if not isinstance(prof, dict) or not isinstance(prof.get("sensor"), dict):
        raise ValueError("not a camera look profile (no 'sensor' section)")
    cur = cfg.sensor.model_dump(mode="json")
    cur.update(prof["sensor"])
    cur["enabled"] = True                   # a look can only show with sensor effects on
    sensor = SensorConfig.model_validate(cur)
    tuned_light = LightingConfig.model_validate(prof["lighting"]) if prof.get("lighting") else None
    if "ae_metering" not in prof["sensor"] and sensor.auto_exposure:
        # profile from before the metering mode was stored: its target meant "mean brightness of the frame".
        # Convert it to the equivalent grey-card target, so parts get the brightness the profile was tuned to
        # and the table / background does not shift it.
        sensor.ae_metering = "light"
        sensor.ae_target = light_metering_target(sensor.exposure, tuned_light or cfg.scene.lighting, sensor.tonemap)
    cfg.sensor = sensor
    if lighting and tuned_light is not None:   # exposure was tuned for this lighting
        keep = cfg.scene.lighting              # ... but shadow settings are not part of a camera look
        tuned_light.shadows, tuned_light.shadow_map_size = keep.shadows, keep.shadow_map_size
        tuned_light.relight_per_view = keep.relight_per_view
        cfg.scene.lighting = tuned_light
    log = [f"sensor: exposure {cfg.sensor.exposure.min:.2f}-{cfg.sensor.exposure.max:.2f}, white balance bias "
           f"{np.round(cfg.sensor.white_balance_bias, 3).tolist()}, noise <={cfg.sensor.gaussian_noise.max:.4f}, "
           f"defocus p={cfg.sensor.defocus_prob:.2f}, saturation {cfg.sensor.saturation.min:.2f}-{cfg.sensor.saturation.max:.2f}"]
    if lighting and prof.get("lighting"):
        log.append("lighting: as used when the profile was tuned")
    cfg.look_profile = path or prof.get("name", "")
    return log


def fix_legacy_look(cfg: ProjectConfig) -> bool:
    """Projects saved while an old look profile's 'frame brightness' target was wrongly used as a grey-card
    target: convert the target (once). Returns True if the project was corrected."""
    s = cfg.sensor
    if not (cfg.look_profile and s.auto_exposure and s.ae_metering == "light"):
        return False
    name = cfg.look_profile.replace("\\", "/").rsplit("/", 1)[-1]
    try:
        prof = load_profile(str(LOOKS_DIR / name))
    except Exception:
        return False
    old = prof["sensor"].get("ae_target")
    if "ae_metering" in prof["sensor"] or not old:
        return False
    if abs(s.ae_target.min - old["min"]) > 1e-9 or abs(s.ae_target.max - old["max"]) > 1e-9:
        return False        # the user set their own target: leave it alone
    from .config import LightingConfig
    light = LightingConfig.model_validate(prof["lighting"]) if prof.get("lighting") else cfg.scene.lighting
    s.ae_target = light_metering_target(s.exposure, light, s.tonemap)
    return True


# --------------------------------------------------------------------------- background option (separate)
def measure_surface_colors(folders, log: Callable[[str], None] = print) -> List[Dict]:
    """Dominant table / work-surface colours of real photos (annotated objects masked)."""
    _, palette = measure_real(folders, log=log)
    return palette


def apply_surface_colors(cfg: ProjectConfig, palette: List[Dict]) -> str:
    """Use measured surface colours as table colours (a background option, independent of the camera look)."""
    g = cfg.scene.ground
    g.palette = [p["color"] for p in palette]
    if g.texture_mode not in ("shadow_catcher", "image_folder"):
        g.texture_mode = "palette"
    return f"table colours: {len(g.palette)} measured surface colours (ground mode {g.texture_mode})"


def _sat(c) -> float:
    mx, mn = max(c), min(c)
    return 0.0 if mx <= 1e-6 else (mx - mn) / mx


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "look"


def report(prof: Dict) -> str:
    r, s = prof["real_stats"], prof["synthetic_stats"]
    lines = [f"{'metric':12s} {'real':>9s} {'synthetic':>10s}"]
    for k in METRICS:
        lines.append(f"{k:12s} {r[k]['mean']:9.4g} {s[k]['mean']:10.4g}")
    return "\n".join(lines)
