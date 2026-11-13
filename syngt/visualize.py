"""GT overlay drawing (used by preview, dataset browser and CLI preview)."""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import project

DEFAULT_LAYERS = {"mask": True, "bbox_visib": True, "bbox_amodal": False, "obb": False, "box3d": True,
                  "axes": False, "keypoints": False, "labels": True, "ignored": True}

_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def color_for(i: int) -> tuple:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    v = d > 0
    if not v.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(d[v], 1), np.percentile(d[v], 99)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    c = cv2.applyColorMap((255 - n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[~v] = 0
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def instance_colormap(inst: np.ndarray) -> np.ndarray:
    out = np.zeros((*inst.shape, 3), np.uint8)
    for i in np.unique(inst):
        if i == 0:
            continue
        out[inst == i] = color_for(int(i))
    return out


def normals_vis(n: np.ndarray) -> np.ndarray:
    return np.clip((n + 1) * 127.5, 0, 255).astype(np.uint8)


def draw_overlay(rgb: np.ndarray, anns: List[dict], class_names: Optional[Dict[int, str]] = None,
                 layers: Optional[dict] = None, K: Optional[np.ndarray] = None, dist=None,
                 inst_mask: Optional[np.ndarray] = None, thresholds: Optional[dict] = None) -> np.ndarray:
    L = dict(DEFAULT_LAYERS)
    L.update(layers or {})
    img = np.ascontiguousarray(rgb[..., :3]).copy()
    H, W = img.shape[:2]
    lw = max(1, int(round(max(H, W) / 700)))
    fs = max(0.4, max(H, W) / 1800)
    th = thresholds or {}

    def ignored(a):
        return (a.get("visib_fract", 1) < th.get("min_visib_fract", 0) or a.get("px_count_visib", 1) < th.get("min_px_visib", 0)
                or a.get("truncation", 0) > th.get("max_truncation", 1))

    if L["mask"]:
        over = img.copy()
        for a in anns:
            col = color_for(a["inst_id"])
            m = None
            if inst_mask is not None:
                m = inst_mask == a["inst_id"]
            elif a.get("rle_visib"):
                from .annotations import rle_decode
                m = rle_decode(a["rle_visib"])
            if m is not None and m.shape == (H, W):
                over[m] = col
        img = cv2.addWeighted(img, 0.55, over, 0.45, 0)
    for a in anns:
        ig = ignored(a)
        if ig and not L["ignored"]:
            continue
        col = (128, 128, 128) if ig else color_for(a["inst_id"])
        if L["bbox_amodal"] and a.get("bbox_obj"):
            x, y, w, h = a["bbox_obj"]
            _dashed_rect(img, (int(x), int(y)), (int(x + w), int(y + h)), col, lw)
        if L["bbox_visib"] and a.get("bbox_visib"):
            x, y, w, h = a["bbox_visib"]
            cv2.rectangle(img, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), col, lw)
        if L["obb"]:
            ob = a.get("obb_visib")
            if ob:
                cv2.polylines(img, [np.round(np.asarray(ob["corners"])).astype(np.int32)], True, (255, 255, 0) if not ig else col, lw)
        if L["box3d"] and a.get("bbox3d_corners_px"):
            c = np.asarray(a["bbox3d_corners_px"])
            if np.all(np.isfinite(c)) and np.abs(c).max() < 1e5:
                for i, j in _EDGES:
                    cv2.line(img, tuple(np.round(c[i]).astype(int)), tuple(np.round(c[j]).astype(int)), col, lw, cv2.LINE_AA)
        if L["axes"] and K is not None:
            T = np.asarray(a["T_cam_obj"])
            ln = 0.03
            P = np.array([[0, 0, 0], [ln, 0, 0], [0, ln, 0], [0, 0, ln]]) @ T[:3, :3].T + T[:3, 3]
            if np.all(P[:, 2] > 0):
                uv = project(np.asarray(K), P, dist).round().astype(int)
                for k, cc in zip((1, 2, 3), ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                    cv2.line(img, tuple(uv[0]), tuple(uv[k]), cc, lw + 1, cv2.LINE_AA)
        if L["keypoints"]:
            for u, v, vis in a.get("keypoints", []):
                if vis > 0:
                    cv2.circle(img, (int(u), int(v)), lw + 2, (0, 255, 0) if vis == 2 else (255, 80, 0), -1)
        if L["labels"] and a.get("bbox_visib"):
            x, y = a["bbox_visib"][:2]
            name = (class_names or {}).get(a["obj_id"], a.get("class_name", str(a["obj_id"])))
            txt = f"{name} v{a['visib_fract']:.2f}" + (" t%.2f" % a["truncation"] if a.get("truncation", 0) > 0.01 else "")
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            y0 = max(int(y) - 4, tht + 4)
            cv2.rectangle(img, (int(x), y0 - tht - 4), (int(x) + tw + 4, y0 + 2), col, -1)
            cv2.putText(img, txt, (int(x) + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _dashed_rect(img, p0, p1, col, lw, dash=8):
    x0, y0 = p0
    x1, y1 = p1
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        a, b = np.array(a, float), np.array(b, float)
        L = np.linalg.norm(b - a)
        n = max(1, int(L // dash))
        for k in range(0, n, 2):
            s = a + (b - a) * k / n
            e = a + (b - a) * min(k + 1, n) / n
            cv2.line(img, tuple(s.astype(int)), tuple(e.astype(int)), col, lw)
