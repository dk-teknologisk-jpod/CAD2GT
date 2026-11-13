"""Procedural textures, image-folder textures/backgrounds and phone-photo rectification."""
from __future__ import annotations

import colorsys
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    return [str(p) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMG_EXTS]


def file_stamp(path: str) -> str:
    """Changes whenever the file is replaced (used in cache keys: an overwritten texture must be reloaded)."""
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}-{st.st_size}"
    except OSError:
        return "missing"


def load_image_rgb(path: str, max_side: int = 4096) -> np.ndarray:
    return _load_image_rgb(path, max_side, file_stamp(path))


@lru_cache(maxsize=8)       # decoded photos are up to 4096^2 x 3 bytes (48 MB) each, per process
def _load_image_rgb(path: str, max_side: int, stamp: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image {path}")
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def texture_physical_size(path: str) -> Optional[Tuple[float, float]]:
    """Width/height in metres from a sidecar json written by the rectification tool."""
    j = Path(path).with_suffix(".json")
    if j.exists():
        try:
            d = json.loads(j.read_text())
            return float(d["width_m"]), float(d["height_m"])
        except Exception:
            return None
    return None


def random_color(rng: np.random.Generator, s_range=(0.0, 1.0), v_range=(0.1, 1.0)) -> np.ndarray:
    h = rng.uniform()
    s = rng.uniform(*s_range)
    v = rng.uniform(*v_range)
    return np.array(colorsys.hsv_to_rgb(h, s, v))


def _fractal_noise(rng: np.random.Generator, size: int, octaves: int = 6, persistence: float = 0.55) -> np.ndarray:
    out = np.zeros((size, size), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = 2 ** (o + 2)
        g = rng.random((n, n)).astype(np.float32)
        g = np.pad(g, 1, mode="wrap")  # tileable
        up = cv2.resize(g, (size + 2 * size // n, size + 2 * size // n), interpolation=cv2.INTER_CUBIC)
        off = size // n
        out += amp * up[off:off + size, off:off + size]
        tot += amp
        amp *= persistence
    out /= tot
    out -= out.min()
    out /= out.max() + 1e-6
    return out


def desaturate(img_u8: np.ndarray, sat_max: float) -> np.ndarray:
    """Limit the mean saturation of an RGB uint8 image to ~sat_max (blend towards grey)."""
    if sat_max >= 0.999:
        return img_u8
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].mean() / 255.0
    if s <= sat_max:
        return img_u8
    f = sat_max / max(s, 1e-6)
    g = img_u8.astype(np.float32).mean(2, keepdims=True)
    return np.clip(g + (img_u8.astype(np.float32) - g) * f, 0, 255).astype(np.uint8)


def procedural_texture(rng: np.random.Generator, size: int = 1024) -> np.ndarray:
    """uint8 RGB texture: noise / wood / checker / speckle / stripes / brushed metal."""
    kind = rng.choice(["noise", "wood", "checker", "speckle", "stripes", "brushed", "plain"])
    c1, c2 = random_color(rng), random_color(rng)
    if kind == "noise":
        t = _fractal_noise(rng, size, persistence=rng.uniform(0.4, 0.7))
    elif kind == "wood":
        n = _fractal_noise(rng, size, 5)
        y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
        t = 0.5 + 0.5 * np.sin((y * rng.uniform(15, 60) + n * rng.uniform(2, 6)) * 2 * np.pi)
        c1 = np.array([rng.uniform(0.35, 0.7), rng.uniform(0.2, 0.45), rng.uniform(0.05, 0.25)])
        c2 = c1 * rng.uniform(0.5, 0.8)
    elif kind == "checker":
        k = int(rng.integers(4, 24))
        yy, xx = np.mgrid[0:size, 0:size]
        t = (((yy * k // size) + (xx * k // size)) % 2).astype(np.float32)
    elif kind == "speckle":
        t = (rng.random((size, size)) > rng.uniform(0.6, 0.97)).astype(np.float32)
        t = cv2.GaussianBlur(t, (0, 0), rng.uniform(0.5, 2.0))
        t = 0.6 * t / (t.max() + 1e-6) + 0.4 * _fractal_noise(rng, size, 4)
    elif kind == "stripes":
        k = rng.uniform(4, 40)
        x = np.linspace(0, 1, size, dtype=np.float32)[None, :]
        t = np.repeat((np.sin(x * k * 2 * np.pi) > 0).astype(np.float32), size, 0)
    elif kind == "brushed":
        n = rng.random((size, size)).astype(np.float32)
        t = cv2.blur(n, (int(rng.integers(30, 120)), 1))
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        c1 = np.full(3, rng.uniform(0.4, 0.8))
        c2 = c1 * 0.85
    else:
        t = 0.15 * _fractal_noise(rng, size, 4)
        c2 = c1 * 0.9
    img = c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def augment_texture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    hsv[..., 2] *= rng.uniform(0.75, 1.2)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


# --------------------------------------------------------------------------- #
# Backgrounds (2D backdrop behind everything rendered)
# --------------------------------------------------------------------------- #
def make_background_u8(mode: str, w: int, h: int, rng: np.random.Generator, images: List[str],
                       random_crop: bool = True, flip: bool = True, saturation_max: float = 1.0,
                       crop_scale: Tuple[float, float] = (0.4, 1.0)) -> Tuple[Optional[np.ndarray], str]:
    """Returns (uint8 sRGB HxWx3 or None for transparent, mode_used). Brightness is applied by the caller."""
    if mode == "mixed":
        choices = ["solid_random", "gradient", "noise"] + (["image_folder"] * 3 if images else [])
        mode = str(rng.choice(choices))
    if mode == "image_folder" and not images:
        mode = "noise"
    if mode == "transparent":
        return None, mode
    if mode == "solid_random":
        img = np.empty((h, w, 3), np.uint8)
        img[:] = (random_color(rng) * 255 + 0.5).astype(np.uint8)
    elif mode == "gradient":
        c1, c2 = random_color(rng), random_color(rng)
        a = rng.uniform(0, 2 * np.pi)
        sw, sh = max(2, w // 8), max(2, h // 8)          # smooth -> compute small, upsample
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        t = (np.cos(a) * xx / sw + np.sin(a) * yy / sh)
        t = (t - t.min()) / (t.max() - t.min() + 1e-6)
        small = (c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]) * 255
        img = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    elif mode == "noise":
        img = cv2.resize(procedural_texture(rng, 512), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        src = load_image_rgb(images[int(rng.integers(len(images)))])
        sh, sw = src.shape[:2]
        if random_crop:
            s = rng.uniform(*sorted(crop_scale))
            ar = w / h
            cw = int(min(sw, sh * ar) * s)
            ch = int(cw / ar)
            cw, ch = max(cw, 8), max(ch, 8)
            x0 = int(rng.integers(0, max(1, sw - cw + 1)))
            y0 = int(rng.integers(0, max(1, sh - ch + 1)))
            src = src[y0:y0 + ch, x0:x0 + cw]
        img = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA if src.shape[1] > w else cv2.INTER_LINEAR)
        if flip and rng.random() < 0.5:
            img = img[:, ::-1]
        return np.ascontiguousarray(img), mode   # photos: keep their real colours
    return np.ascontiguousarray(desaturate(img, saturation_max)), mode


def background_to_linear(img_u8: np.ndarray, brightness: float) -> np.ndarray:
    lut = np.power(np.clip(np.arange(256, dtype=np.float32) / 255.0 * brightness, 0, 1), 2.2).astype(np.float32)
    return lut[img_u8]


def make_background(mode: str, w: int, h: int, rng: np.random.Generator, images: List[str],
                    random_crop: bool = True, flip: bool = True, brightness: float = 1.0) -> Tuple[Optional[np.ndarray], str]:
    """Returns (linear float RGB HxWx3 or None for transparent, mode_used)."""
    img, mode = make_background_u8(mode, w, h, rng, images, random_crop, flip)
    return (None if img is None else background_to_linear(img, brightness)), mode


# --------------------------------------------------------------------------- #
# Phone photo -> metric ground texture
# --------------------------------------------------------------------------- #
def rectify_photo(img_rgb: np.ndarray, pts: np.ndarray, width_mm: float, height_mm: float,
                  px_per_mm: float = 2.0, make_tileable: bool = True, crop_inset: float = 0.0) -> np.ndarray:
    """Warp the quadrilateral ``pts`` (TL, TR, BR, BL pixel coords) of a photo of a planar surface
    to a fronto-parallel, metrically scaled texture (width_mm x height_mm)."""
    W = max(8, int(round(width_mm * px_per_mm)))
    H = max(8, int(round(height_mm * px_per_mm)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(pts, np.float32), dst)
    out = cv2.warpPerspective(img_rgb, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if crop_inset > 0:
        dx, dy = int(W * crop_inset), int(H * crop_inset)
        out = out[dy:H - dy, dx:W - dx]
    if make_tileable:
        out = make_seamless(out)
    return out


def flatten_illumination(img_rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove low-frequency shading (phone flash / vignetting) so the texture is albedo-like."""
    f = img_rgb.astype(np.float32) + 1.0
    sigma = max(img_rgb.shape[:2]) / 6
    low = cv2.GaussianBlur(f, (0, 0), sigma)
    mean = low.reshape(-1, 3).mean(0)
    out = f / low * mean
    out = f * (1 - strength) + out * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def make_seamless(img: np.ndarray, blend: float = 0.15) -> np.ndarray:
    """Cross-fade the image with its half-period shifted copy to remove tile seams."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    shifted = np.roll(np.roll(f, h // 2, 0), w // 2, 1)
    wy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32) / (h * blend)
    wx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32) / (w * blend)
    m = np.clip(np.minimum(wy[:, None], wx[None, :]), 0, 1)[..., None]
    return np.clip(f * m + shifted * (1 - m), 0, 255).astype(np.uint8)


def save_texture(img_rgb: np.ndarray, path: str, width_mm: float, height_mm: float, source: str = "") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(p.suffix or ".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))[1].tofile(str(p))
    p.with_suffix(".json").write_text(json.dumps({"width_m": width_mm / 1000.0, "height_m": height_mm / 1000.0,
                                                  "source": source}, indent=2))
