"""GPU sensor post-processing must match the CPU reference implementation.

Run:  .venv\\Scripts\\python tests\\test_gpu_postprocess.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_consistency import base_cfg  # noqa: E402
from syngt.cameras import camera_from_preset  # noqa: E402
from syngt.config import R  # noqa: E402
from syngt.scene import SceneSampler  # noqa: E402


def frame(cfg, gpu: bool, sampler=None):
    cfg.render.gpu_postprocess = gpu
    s = sampler or SceneSampler(cfg)
    s.cfg = cfg
    st = s.sample_scene(1)
    return s.render_view(st, 0, np.random.default_rng(5))[0], s


def compare(label: str, mutate, noise=False) -> bool:
    cfg = base_cfg()
    s = cfg.sensor
    s.defocus_prob = s.motion_blur_prob = 0.0
    s.jpeg_quality = R(100, 100)
    if not noise:
        s.gaussian_noise = R(0, 0)
        s.shot_noise = R(0, 0)
    mutate(cfg)
    a, smp = frame(cfg, True)
    b, _ = frame(cfg, False, smp)
    d = np.abs(a.rgb.astype(np.int16) - b.rgb.astype(np.int16))
    if noise:
        ok = abs(float(a.rgb.std()) - float(b.rgb.std())) < 2.0 and abs(float(a.rgb.mean()) - float(b.rgb.mean())) < 1.0
        print(f"[{'OK ' if ok else 'FAIL'}] {label:34s} mean gpu/cpu {a.rgb.mean():.2f}/{b.rgb.mean():.2f}  "
              f"std {a.rgb.std():.2f}/{b.rgb.std():.2f}")
    else:
        ok = d.mean() < 0.6 and np.percentile(d, 99.9) <= 6
        print(f"[{'OK ' if ok else 'FAIL'}] {label:34s} mean|d|={d.mean():.3f}  p99.9={np.percentile(d, 99.9):.0f}  max={d.max()}")
    return ok


def main():
    def dist(cfg):
        c = camera_from_preset("Intel RealSense D435 / D435i", "color", 1280, 720)
        c.dist = [-0.28, 0.09, 0.001, -0.0015, -0.012]
        cfg.cameras = [c]

    def defocus(cfg):
        cfg.sensor.defocus_prob = 1.0

    def motion(cfg):
        cfg.sensor.motion_blur_prob = 1.0

    def alpha(cfg):
        cfg.scene.background.mode = "transparent"

    def photo(cfg):
        cfg.scene.background.mode = "image_folder"
        cfg.scene.background.image_dir = str(ROOT / "backgrounds" / "workcell_plates")
        cfg.scene.ground.texture_mode = "shadow_catcher"
        cfg.sensor.defocus_prob = 1.0

    r = [compare("plain (crop canvas)", lambda c: None), compare("lens distortion (remap)", dist),
         compare("real photo bg + shadow catcher + blur", photo),
         compare("defocus blur", defocus), compare("motion blur", motion), compare("transparent background", alpha),
         compare("noise statistics", lambda c: None, noise=True)]
    # timing at 4K
    cfg = base_cfg()
    cfg.cameras = [camera_from_preset("Orbbec Femto Bolt / Mega (NFOV)", "color", 3840, 2160)]
    smp = SceneSampler(cfg)
    st = smp.sample_scene(0)
    print("GL renderer:", smp.renderer.info["GL_RENDERER"])
    for gpu in (False, True):
        cfg.render.gpu_postprocess = gpu
        smp.render_view(st, 0, np.random.default_rng(0))
        t = time.time()
        for v in range(3):
            smp.render_view(st, v % len(st.views), np.random.default_rng(v))
        print(f"4K frame, sensor model on {'GPU' if gpu else 'CPU'}: {(time.time() - t) / 3 * 1000:.0f} ms")
    print("ALL OK" if all(r) else "SOME CHECKS FAILED")
    return 0 if all(r) else 1


if __name__ == "__main__":
    sys.exit(main())
