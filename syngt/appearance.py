"""Measure the look of real camera images and derive matching sensor / lighting settings.

The statistics are deliberately simple and robust (no reference objects needed):
brightness, contrast, saturation, colour cast, noise level (MAD of a high-pass on flat areas),
sharpness (variance of the Laplacian) and the fraction of clipped highlights.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List

import cv2
import numpy as np

from .config import R, ProjectConfig

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str, limit: int = 60) -> List[str]:
    # glob.escape: folder names like "photos [cam1]" contain glob characters
    fs = sorted(f for f in glob.glob(os.path.join(glob.escape(folder), "**", "*"), recursive=True)
                if f.lower().endswith(IMG_EXT))
    if len(fs) > limit:
        fs = fs[:: max(1, len(fs) // limit)][:limit]
    return fs


def image_stats(img_rgb: np.ndarray) -> Dict[str, float]:
    img = img_rgb.astype(np.float32) / 255.0
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    # noise: residual of a 3x3 median on low-gradient pixels
    smooth = cv2.GaussianBlur(g, (0, 0), 1.0)
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    flat = grad < np.percentile(grad, 40)
    resid = (g - smooth)[flat]
    noise = float(resid.std()) / 255.0 if resid.size else 0.0  # high-frequency residual on flat areas
    mean_rgb = img.reshape(-1, 3).mean(0)
    return {
        "brightness": float(g.mean() / 255.0),
        "contrast": float(g.std() / 255.0),
        "saturation": float(hsv[..., 1].mean() / 255.0),
        "cast_r": float(mean_rgb[0] / max(mean_rgb[1], 1e-6)),
        "cast_b": float(mean_rgb[2] / max(mean_rgb[1], 1e-6)),
        "noise": noise,
        "sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var()),
        "clipped": float((g >= 250).mean()),
    }


def folder_stats(folder: str, limit: int = 60) -> Dict[str, Dict[str, float]]:
    rows = []
    for f in list_images(folder, limit):
        im = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        rows.append(image_stats(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    if not rows:
        raise RuntimeError(f"no images in {folder}")
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    out["_n"] = {"mean": float(len(rows)), "p10": 0.0, "p90": 0.0}
    return out


def apply_to_config(cfg: ProjectConfig, real: Dict[str, Dict[str, float]],
                    synth: Dict[str, Dict[str, float]] | None = None) -> List[str]:
    """Adjust sensor settings so synthetic images match the measured real ones. Returns a change log."""
    s, log = cfg.sensor, []
    s.enabled = True
    # exposure: scale the current range by the brightness ratio (approximately, in linear space)
    if synth is not None and s.auto_exposure:
        # with auto exposure the fixed exposure is not used: brightness is set by the auto-exposure target
        ratio = real["brightness"]["mean"] / max(synth["brightness"]["mean"], 1e-3)
        lo = float(np.clip(s.ae_target.min * ratio, 0.05, 0.9))
        hi = float(np.clip(s.ae_target.max * ratio, lo, 0.95))
        s.ae_target = R(lo, hi)
        log.append(f"auto-exposure target x{ratio:.2f} -> {lo:.2f}-{hi:.2f} (brightness real "
                   f"{real['brightness']['mean']:.2f} vs synthetic {synth['brightness']['mean']:.2f})")
    elif synth is not None:
        ratio = (real["brightness"]["mean"] / max(synth["brightness"]["mean"], 1e-3)) ** 2.2
        lo = float(np.clip(s.exposure.min * ratio, 0.01, 100.0))
        hi = float(np.clip(s.exposure.max * ratio, lo, 100.0))
        s.exposure = R(lo, hi, "loguniform")
        log.append(f"exposure x{ratio:.2f} -> {lo:.2f}-{hi:.2f} (brightness real {real['brightness']['mean']:.2f} "
                   f"vs synthetic {synth['brightness']['mean']:.2f})")
    # read noise: measured noise is after gamma; ~linear-space sigma is smaller in bright areas -> use as is
    n = real["noise"]
    s.gaussian_noise = R(max(0.0, n["p10"] * 0.6), max(n["p90"] * 1.2, 0.003))
    s.shot_noise = R(0.0, max(n["p90"], 0.004))
    log.append(f"read noise {s.gaussian_noise.min:.4f}-{s.gaussian_noise.max:.4f} (measured {n['mean'] * 255:.1f} grey levels)")
    # colour cast -> white balance jitter wide enough to cover the measured casts
    cast = max(abs(real["cast_r"]["p10"] - 1), abs(real["cast_r"]["p90"] - 1),
               abs(real["cast_b"]["p10"] - 1), abs(real["cast_b"]["p90"] - 1))
    s.white_balance_jitter = float(np.clip(cast * 1.2, 0.03, 0.12))
    log.append(f"white-balance jitter +-{s.white_balance_jitter:.2f} (measured R/G {real['cast_r']['mean']:.2f}, "
               f"B/G {real['cast_b']['mean']:.2f})")
    # saturation / contrast relative to the synthetic set
    if synth is not None:
        sr = real["saturation"]["mean"] / max(synth["saturation"]["mean"], 1e-3)
        cr = real["contrast"]["mean"] / max(synth["contrast"]["mean"], 1e-3)
        s.saturation = R(float(np.clip(sr * 0.85, 0.2, 2)), float(np.clip(sr * 1.15, 0.3, 2.5)))
        s.contrast = R(float(np.clip(cr * 0.9, 0.4, 2)), float(np.clip(cr * 1.1, 0.5, 2.5)))
        log.append(f"saturation {s.saturation.min:.2f}-{s.saturation.max:.2f}, contrast {s.contrast.min:.2f}-{s.contrast.max:.2f}")
        # blur: real images much softer than synthetic -> more / stronger defocus
        shr = real["sharpness"]["mean"] / max(synth["sharpness"]["mean"], 1e-3)
        if shr < 0.8:
            s.defocus_prob = float(np.clip(1.0 - shr, 0.3, 0.9))
            s.defocus_sigma = R(0.4, float(np.clip(1.5 / max(shr, 0.2), 0.8, 3.0)))
            log.append(f"defocus prob {s.defocus_prob:.2f}, sigma {s.defocus_sigma.min:.1f}-{s.defocus_sigma.max:.1f} px "
                       f"(real images {1 / max(shr, 1e-3):.1f}x softer)")
        elif shr > 1.25:  # synthetic softer than real -> less blur
            s.defocus_prob = float(np.clip(s.defocus_prob / shr, 0.05, 0.9))
            s.defocus_sigma = R(0.3, float(max(0.5, s.defocus_sigma.max / np.sqrt(shr))))
            log.append(f"defocus prob {s.defocus_prob:.2f}, sigma {s.defocus_sigma.min:.1f}-{s.defocus_sigma.max:.1f} px "
                       f"(real images {shr:.1f}x sharper)")
    return log


def synthetic_stats(cfg: ProjectConfig, n_frames: int = 8, seed0: int = 1000) -> Dict[str, Dict[str, float]]:
    """Render a few frames with the current settings and measure them like the real photos."""
    from .scene import SceneSampler
    s = SceneSampler(cfg)
    rows = []
    k = 0
    while len(rows) < n_frames and k < n_frames * 3:
        st = s.sample_scene(seed0 + k)
        fr = s.render_view(st, 0, np.random.default_rng(k))[0]
        rows.append(image_stats(fr.rgb[..., :3]))
        k += 1
    out = {}
    for key in rows[0]:
        v = np.array([r[key] for r in rows])
        out[key] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    return out


def compare_table(real: Dict, synth: Dict) -> str:
    rows = [("brightness", "{:.2f}"), ("contrast", "{:.3f}"), ("saturation", "{:.3f}"), ("cast_r", "{:.2f}"),
            ("cast_b", "{:.2f}"), ("noise", "{:.4f}"), ("sharpness", "{:.0f}")]
    out = ["metric        real      synthetic"]
    for k, f in rows:
        out.append(f"{k:12s} {f.format(real[k]['mean']):>8s}  {f.format(synth[k]['mean']):>10s}")
    return "\n".join(out)
