"""GT self-consistency checks across placement modes / camera models.

Run:  .venv\\Scripts\\python tests\\test_consistency.py
Checks for every annotated instance that the CAD vertices, transformed by the GT pose and projected with
the GT intrinsics (+distortion), fall inside the rendered amodal silhouette, and that the visible depth
lies within the object's z-range. Any mismatch indicates a convention error between renderer and GT.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syngt.annotations import rle_decode  # noqa: E402
from syngt.cameras import camera_from_preset  # noqa: E402
from syngt.config import IntRange, ObjectModel, ProjectConfig  # noqa: E402
from syngt.geometry import look_at_cv, project, transform_points  # noqa: E402
from syngt.scene import SceneSampler  # noqa: E402


def base_cfg() -> ProjectConfig:
    cfg = ProjectConfig()
    for i, n in enumerate(["gear", "bracket", "flange", "hexnut"], 1):
        cfg.objects.models.append(ObjectModel(path=str(ROOT / "examples" / "models" / f"{n}.stl"), name=n,
                                              class_id=i, unit="mm", count=IntRange(min=1, max=2)))
    cfg.objects.distractors.enabled = True
    cfg.viewpoints.views_per_scene = 2
    return cfg


def check(cfg: ProjectConfig, label: str, scenes=2) -> bool:
    t0 = time.time()
    s = SceneSampler(cfg)
    fracs, dz, n = [], [], 0
    for si in range(scenes):
        st = s.sample_scene(si)
        assets = {a["inst_id"]: a for a in st.annotated}
        for v in range(len(st.views)):
            for fr in s.render_view(st, v, np.random.default_rng(v)):
                H, W = fr.inst.shape
                for a in fr.annotations:
                    A = assets[a["inst_id"]]
                    T = np.asarray(a["T_cam_obj"])
                    P = transform_points(T, A["asset"].vertices.astype(np.float64) * A["scale"])
                    if np.any(P[:, 2] <= 0.01):
                        continue
                    uv = project(fr.K, P, fr.dist)
                    ins = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
                    if ins.sum() < 5 or a["rle_amodal"] is None:
                        continue
                    full = cv2.dilate(rle_decode(a["rle_amodal"]).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    u = np.round(uv[ins]).astype(int)
                    fracs.append(full[u[:, 1], u[:, 0]].mean())
                    n += 1
                    vis = fr.inst == a["inst_id"]
                    if vis.sum() > 30:
                        z = fr.depth_gt[vis]
                        dz.append(np.mean((z >= P[:, 2].min() - 1e-3) & (z <= P[:, 2].max() + 1e-3)))
    ok = n > 0 and np.min(fracs) > 0.97 and (not dz or np.min(dz) > 0.98)
    print(f"[{'OK ' if ok else 'FAIL'}] {label:38s} inst={n:3d} vert-in-mask min={np.min(fracs) if fracs else 0:.4f} "
          f"mean={np.mean(fracs) if fracs else 0:.4f}  depth-ok min={np.min(dz) if dz else 1:.4f}  ({time.time() - t0:.1f}s)")
    return ok


def main() -> int:
    results = []
    for mode in ["physics_drop", "physics_bin", "stable_separated", "free_floating"]:
        cfg = base_cfg()
        cfg.placement.mode = mode
        if mode == "free_floating":
            cfg.scene.ground.enabled = False
            cfg.viewpoints.elevation_deg.min = -60
        results.append(check(cfg, f"placement={mode}"))

    cfg = base_cfg()
    cam = camera_from_preset("Intel RealSense D435 / D435i", "color", 1280, 720)
    cam.dist = [-0.28, 0.09, 0.001, -0.0015, -0.012]
    cfg.cameras = [cam]
    results.append(check(cfg, "distorted camera (k1=-0.28)"))

    cfg = base_cfg()
    c0 = camera_from_preset("Orbbec Gemini 335 / 336", "color", 1280, 800, name="a")
    c1 = camera_from_preset("Orbbec Femto Bolt / Mega (NFOV)", "color", 1920, 1080, name="b")
    c0.world_pose = look_at_cv([0.0, -0.45, 0.45], [0, 0, 0]).tolist()
    c1.world_pose = look_at_cv([0.4, 0.2, 0.5], [0, 0, 0]).tolist()
    cfg.cameras = [c0, c1]
    cfg.viewpoints.mode = "fixed"
    cfg.viewpoints.fixed_jitter_rotation_deg = 2.0
    results.append(check(cfg, "fixed calibrated 2-camera rig"))

    cfg = base_cfg()
    cfg.cameras = [camera_from_preset("Intel RealSense D455 / D457", "color", 1280, 800, name="x"),
                   camera_from_preset("Intel RealSense D405", "color", 848, 480, name="y")]
    cfg.viewpoints.rig_mode = "independent"
    cfg.viewpoints.mode = "sphere_fibonacci"
    cfg.scene.background.mode = "transparent"
    results.append(check(cfg, "independent cams, fibonacci, alpha bg"))
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
