"""Image formation: padded/undistorted render canvas -> (distorted) sensor image, sensor and depth noise."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig, DepthConfig, SensorConfig


class CameraCanvas:
    """Rendering happens on a larger pinhole canvas (padding for amodal boxes, and to cover the
    undistorted extent of a distorted camera). ``to_image`` maps canvas buffers to sensor pixels."""

    _map_cache: dict = {}

    def __init__(self, K: np.ndarray, width: int, height: int, dist, padding: float = 0.15, max_scale: float = 2.5):
        self.K = np.asarray(K, np.float64)
        self.W, self.H = int(width), int(height)
        self.dist = np.asarray(dist if dist is not None else np.zeros(5), np.float64).reshape(-1)
        self.distorted = bool(np.any(np.abs(self.dist) > 1e-12))
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        if not self.distorted:
            px, py = int(round(padding * self.W)), int(round(padding * self.H))
            self.px, self.py = px, py
            self.Wr, self.Hr = self.W + 2 * px, self.H + 2 * py
            self.Kr = self.K.copy()
            self.Kr[0, 2] += px
            self.Kr[1, 2] += py
            self.map = None
        else:
            key = (tuple(np.round(self.K.reshape(-1), 6)), self.W, self.H, tuple(np.round(self.dist, 9)), padding)
            if key in CameraCanvas._map_cache:
                self.Kr, self.Wr, self.Hr, self.map = CameraCanvas._map_cache[key]
            else:
                uu, vv = np.meshgrid(np.arange(self.W, dtype=np.float64), np.arange(self.H, dtype=np.float64))
                pts = np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2)
                crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9)
                if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
                    und = cv2.undistortPointsIter(pts, self.K, self.dist, None, None, crit)
                else:  # OpenCV 5.x merged the iterative variant
                    und = cv2.undistortPoints(pts, self.K, self.dist, R=None, P=None, criteria=crit)
                und = und.reshape(-1, 2)
                xn, yn = und[:, 0], und[:, 1]
                # limit canvas growth for extreme distortion
                lim_x = (self.W * max_scale / 2) / fx
                lim_y = (self.H * max_scale / 2) / fy
                cxn, cyn = (self.W / 2 - cx) / fx, (self.H / 2 - cy) / fy
                xn = np.clip(xn, -lim_x + cxn, lim_x + cxn)
                yn = np.clip(yn, -lim_y + cyn, lim_y + cyn)
                pad_x, pad_y = padding * self.W, padding * self.H
                x0, x1 = xn.min() * fx - pad_x, xn.max() * fx + pad_x
                y0, y1 = yn.min() * fy - pad_y, yn.max() * fy + pad_y
                self.Wr, self.Hr = int(math.ceil(x1 - x0)) + 1, int(math.ceil(y1 - y0)) + 1
                self.Kr = np.array([[fx, 0, -x0], [0, fy, -y0], [0, 0, 1.0]])
                mx = (xn * fx - x0).reshape(self.H, self.W).astype(np.float32)
                my = (yn * fy - y0).reshape(self.H, self.W).astype(np.float32)
                self.map = (mx, my)
                if len(CameraCanvas._map_cache) > 16:
                    CameraCanvas._map_cache.clear()
                CameraCanvas._map_cache[key] = (self.Kr, self.Wr, self.Hr, self.map)
            self.px = self.py = 0

    # -------------------------------------------------------------- buffers
    def to_image(self, arr: np.ndarray, nearest: bool = False) -> np.ndarray:
        if self.map is None:
            return arr[self.py:self.py + self.H, self.px:self.px + self.W]
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        if arr.dtype == bool:
            return cv2.remap(arr.astype(np.uint8), self.map[0], self.map[1], cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if arr.dtype in (np.int32, np.int64):
            return cv2.remap(arr.astype(np.float32), self.map[0], self.map[1], cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
        return cv2.remap(arr, self.map[0], self.map[1], interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def canvas_to_image_points(self, pts: np.ndarray) -> np.ndarray:
        """Canvas pixel coords (N,2) -> sensor pixel coords (with distortion)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if not self.distorted:
            return pts - np.array([self.px, self.py])
        xn = (pts[:, 0] - self.Kr[0, 2]) / self.Kr[0, 0]
        yn = (pts[:, 1] - self.Kr[1, 2]) / self.Kr[1, 1]
        P = np.stack([xn, yn, np.ones_like(xn)], 1)
        uv, _ = cv2.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    def image_region_in_canvas(self) -> np.ndarray:
        """Bool canvas mask of pixels that are seen by the sensor."""
        m = np.zeros((self.Hr, self.Wr), np.uint8)
        if not self.distorted:
            m[self.py:self.py + self.H, self.px:self.px + self.W] = 1
            return m.astype(bool)
        mx, my = self.map
        border = np.concatenate([np.stack([mx[0], my[0]], 1), np.stack([mx[:, -1], my[:, -1]], 1),
                                 np.stack([mx[-1, ::-1], my[-1, ::-1]], 1), np.stack([mx[::-1, 0], my[::-1, 0]], 1)])
        cv2.fillPoly(m, [np.round(border).astype(np.int32)], 1)
        return m.astype(bool)


# --------------------------------------------------------------------------- #
# Colour sensor
# --------------------------------------------------------------------------- #
def _tonemap(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "filmic":
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    else:
        x = np.clip(x, 0, 1)
    return np.power(x, 1 / 2.2)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Linear motion-blur kernel with the same taps as the GPU pass: L samples at t = i - (L-1)/2 along the
    direction (cos a, -sin a) in image coordinates, splatted bilinearly into an odd, centred kernel."""
    a = np.radians(angle_deg)
    dx, dy = np.cos(a), -np.sin(a)
    r = int(np.ceil((length - 1) / 2.0)) + 1
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    for i in range(int(length)):
        t = i - (length - 1) / 2.0
        x, y = r + dx * t, r + dy * t
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
                if 0 <= yy < k.shape[0] and 0 <= xx < k.shape[1]:
                    k[yy, xx] += wx * wy
    return k / max(float(k.sum()), 1e-6)


def display_brightness(lin: np.ndarray, tonemap: str) -> float:
    return float(_tonemap(np.clip(lin, 0, None), tonemap).mean())


def auto_exposure_gain(small_lin: np.ndarray, target: float, tonemap: str, contrast: float = 1.0,
                       highlight: float = 0.90) -> float:
    """Gain k so that the mean display brightness of k * image equals target (bisection in log space)."""
    x = small_lin.astype(np.float32)
    if float(x.max()) <= 1e-8:
        return 1.0
    lo, hi = -9.0, 9.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if display_brightness(x * np.exp(mid), tonemap) < target:
            lo = mid
        else:
            hi = mid
    k = float(np.exp(0.5 * (lo + hi)))
    # highlight protection (like a real AE): the brightest 1 % must keep visible shading, i.e. stay at
    # `highlight` AFTER the contrast step that follows the tone curve (0.5 + (y - 0.5) * contrast)
    p99 = float(np.percentile(x.max(axis=-1), 99.0))
    if p99 > 1e-8:
        y_max = 0.5 + (highlight - 0.5) / max(contrast, 1e-3)
        k = min(k, inverse_display(y_max, tonemap) / p99)
    return k


def inverse_display(y: float, tonemap: str) -> float:
    """Linear value that the tone curve maps to display value y (0..1)."""
    y = float(np.clip(y, 1e-4, 0.995))
    lo, hi = 0.0, 64.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(_tonemap(np.array([mid], np.float32), tonemap)[0]) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def photo_exposure(bg_u8: np.ndarray, white_radiance: float, tonemap: str, percentile: float = 95.0) -> float:
    """Exposure for parts composited onto a real photo: a white part under the scene light is shown as bright as
    the bright surfaces of the photo (its p95 luminance), so parts and photo share one plausible exposure."""
    small = cv2.resize(bg_u8, (max(1, bg_u8.shape[1] // 8), max(1, bg_u8.shape[0] // 8)), interpolation=cv2.INTER_AREA)
    lum = small.astype(np.float32).mean(axis=2) / 255.0
    target = float(np.percentile(lum, percentile))
    return inverse_display(target, tonemap) / max(white_radiance, 1e-6)


def sample_sensor_params(cfg: SensorConfig, rng: np.random.Generator) -> dict:
    """Draw all random sensor parameters for one image (shared by the GPU and CPU paths)."""
    if not cfg.enabled:
        return {"enabled": False, "tonemap": "srgb", "noise_seed": int(rng.integers(0, 2**31 - 1))}
    p = {"enabled": True, "tonemap": cfg.tonemap}
    p["exposure"] = cfg.exposure.sample(rng)
    p["ae"] = bool(cfg.auto_exposure)
    p["ae_target"] = cfg.ae_target.sample(rng)
    bias = np.asarray((list(cfg.white_balance_bias) + [1.0, 1.0, 1.0])[:3], float)
    p["wb"] = (bias * (1.0 + rng.uniform(-cfg.white_balance_jitter, cfg.white_balance_jitter, 3))).tolist()
    p["vignetting"] = cfg.vignetting.sample(rng)
    p["defocus_sigma"] = cfg.defocus_sigma.sample(rng) if rng.random() < cfg.defocus_prob else 0.0
    if rng.random() < cfg.motion_blur_prob:
        p["motion_blur"] = [max(2, int(round(cfg.motion_blur_length.sample(rng)))), float(rng.uniform(0, 180))]
    else:
        p["motion_blur"] = None
    p["shot_noise"] = cfg.shot_noise.sample(rng)
    p["gaussian_noise"] = cfg.gaussian_noise.sample(rng)
    p["saturation"] = cfg.saturation.sample(rng)
    p["contrast"] = cfg.contrast.sample(rng)
    p["jpeg_quality"] = int(round(cfg.jpeg_quality.sample(rng)))
    p["noise_seed"] = int(rng.integers(0, 2**31 - 1))
    return p


def apply_jpeg(img: np.ndarray, q: int) -> np.ndarray:
    if q >= 100:
        return img
    ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(img[..., :3]), [cv2.IMWRITE_JPEG_QUALITY, max(5, int(q))])
    out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return np.dstack([out, img[..., 3]]) if img.shape[2] == 4 else out


def apply_sensor_cpu(linear_rgb: np.ndarray, p: dict, alpha: Optional[np.ndarray] = None,
                     bg_display: Optional[np.ndarray] = None) -> np.ndarray:
    """CPU reference implementation: linear HDR float RGB -> uint8 (same model as the GPU passes).
    With ``alpha`` the rendered layer is premultiplied. With ``bg_display`` too (a real photo, display space
    0..1) it goes through the sensor model and is composited over the untouched photo -> HxWx3. With ``alpha``
    only (transparent background) the result is HxWx4: straight colour + the (equally blurred) alpha."""
    x = linear_rgb.astype(np.float32)
    a = None if alpha is None else alpha.astype(np.float32).reshape(x.shape[:2])

    def finish(y, a):
        y = np.clip(y, 0, 1)
        if a is not None and bg_display is not None:
            y = y * a[..., None] + bg_display * (1 - a[..., None])
        rgb = (np.clip(y, 0, 1) * 255 + 0.5).astype(np.uint8)
        if a is not None and bg_display is None:
            return np.dstack([rgb, (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)])
        return rgb
    if not p.get("enabled", True):
        return finish(_tonemap(x if a is None else np.where(a[..., None] > 1e-4, x / np.maximum(a, 1e-4)[..., None], 0),
                               "srgb"), a)
    rng = np.random.default_rng(p["noise_seed"])
    h, w = x.shape[:2]
    ae = p.get("ae", False) and bg_display is None   # parts on a real photo keep the profile's exposure
    x = x * ((1.0 if ae else p["exposure"]) * np.asarray(p["wb"])).astype(np.float32)
    v = p["vignetting"]
    if v > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x *= (1 - v * r2 / 2)[..., None]
    if ae:
        small = cv2.resize(x, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_AREA)
        k = auto_exposure_gain(small, p["ae_target"], p["tonemap"], p.get("contrast", 1.0))
        p["ae_gain"] = k
        x = x * np.float32(k)
    if p["defocus_sigma"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["defocus_sigma"])
        if a is not None:
            a = cv2.GaussianBlur(a, (0, 0), p["defocus_sigma"])
    if p["motion_blur"]:
        k = _motion_kernel(*p["motion_blur"])
        x = cv2.filter2D(x, -1, k)
        if a is not None:
            a = cv2.filter2D(a, -1, k)
    if a is not None:
        x = np.where(a[..., None] > 1e-4, x / np.maximum(a, 1e-4)[..., None], 0).astype(np.float32)
    if p["shot_noise"] > 0:
        x = x + rng.standard_normal(x.shape, dtype=np.float32) * np.sqrt(np.clip(x, 0, None)) * p["shot_noise"]
    if p["gaussian_noise"] > 0:
        x = x + rng.standard_normal(x.shape, dtype=np.float32) * p["gaussian_noise"]
    y = _tonemap(np.clip(x, 0, None), p["tonemap"])
    if abs(p["saturation"] - 1) > 1e-3:
        g = y.mean(axis=2, keepdims=True)
        y = g + (y - g) * p["saturation"]
    if abs(p["contrast"] - 1) > 1e-3:
        y = 0.5 + (y - 0.5) * p["contrast"]
    return finish(y, a)


def apply_sensor(linear_rgb: np.ndarray, cfg: SensorConfig, rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """Backwards compatible helper: sample params + CPU path + JPEG."""
    p = sample_sensor_params(cfg, rng)
    img = apply_sensor_cpu(linear_rgb, p)
    return apply_jpeg(img, p.get("jpeg_quality", 100)), p


# --------------------------------------------------------------------------- #
# Depth sensor
# --------------------------------------------------------------------------- #
def apply_depth_noise(depth: np.ndarray, cfg: DepthConfig, fx: float, rng: np.random.Generator) -> np.ndarray:
    z = depth.astype(np.float32).copy()
    valid = z > 0
    if cfg.model != "ideal":
        # depth discontinuities
        k = np.ones((3, 3), np.uint8)
        zmax = cv2.dilate(z, k)
        zmin = cv2.erode(np.where(valid, z, 1e6).astype(np.float32), k)
        edge = valid & ((zmax - zmin) > cfg.edge_threshold * np.maximum(z, 0.1))
        if cfg.model == "stereo":
            sigma = (z ** 2) * cfg.disparity_noise_px / max(fx * cfg.baseline_m, 1e-6)
            n_white = rng.standard_normal(z.shape, dtype=np.float32)
            hh, ww = z.shape
            # spatially correlated component (~3 px correlation length), generated at 1/3 res and upsampled
            n_corr = cv2.resize(rng.standard_normal((hh // 3 + 2, ww // 3 + 2), dtype=np.float32), (ww, hh),
                                interpolation=cv2.INTER_LINEAR) * 1.3
            z = z + sigma * (0.6 * n_white + 0.8 * n_corr)
            # invalidation band around edges (occlusion shadow)
            band = cv2.dilate(edge.astype(np.uint8), np.ones((3, 5), np.uint8)).astype(bool)
            idx = np.flatnonzero(band)
            z.reshape(-1)[idx[rng.random(len(idx), dtype=np.float32) < cfg.edge_dropout]] = 0
        else:  # tof
            sigma = cfg.tof_noise_rel * z + cfg.tof_noise_abs
            z = z + sigma * rng.standard_normal(z.shape, dtype=np.float32)
            # flying pixels: mix of fg/bg depth at edges
            fly = edge & (rng.random(z.shape, dtype=np.float32) < cfg.edge_dropout)
            mix = rng.random(z.shape).astype(np.float32)
            z[fly] = (zmin * mix + zmax * (1 - mix))[fly]
        if cfg.random_dropout > 0:
            holes = (rng.random((max(1, z.shape[0] // 8), max(1, z.shape[1] // 8))) < cfg.random_dropout).astype(np.uint8)
            holes = cv2.resize(holes, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            z[holes] = 0
    z[(z < cfg.min_z) | (z > cfg.max_z) | ~valid] = 0
    return z
