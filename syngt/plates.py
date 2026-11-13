"""Build clean background plates (empty workcell photos) from real, annotated images.

Images taken from the same static camera pose are grouped; annotated objects (YOLO box / polygon .txt,
or COCO annotations.json) are masked out and the per-pixel median over the group recovers the empty scene.
Pixels never free in any image of a group are inpainted. Review the result: unlabelled objects that never
moved remain in the plate.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _read(p: str) -> Optional[np.ndarray]:
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def _coco_polys(folder: str) -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = defaultdict(list)
    for jf in glob.glob(os.path.join(glob.escape(folder), "*.json")):
        try:
            d = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "images" not in d or "annotations" not in d:
            continue
        names = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            fn = names.get(a.get("image_id"))
            if fn is None:
                continue
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg and isinstance(seg[0], list):
                for s in seg:
                    if len(s) >= 6:
                        out[os.path.basename(fn)].append(np.asarray(s, float).reshape(-1, 2))
            elif a.get("bbox"):
                x, y, w, h = a["bbox"]
                out[os.path.basename(fn)].append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float))
    return out


def _yolo_polys(txt: str, w: int, h: int) -> List[np.ndarray]:
    polys = []
    for line in open(txt, encoding="utf-8", errors="ignore"):
        vals = []
        for tok in line.split()[1:]:
            try:
                vals.append(float(tok))
            except ValueError:
                pass  # e.g. group tags
        if len(vals) == 4:  # cx cy w h
            cx, cy, bw, bh = vals
            polys.append(np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                   [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]]) * [w, h])
        elif len(vals) >= 6:
            pts = np.asarray(vals[: len(vals) // 2 * 2], float).reshape(-1, 2)
            polys.append(pts * [w, h])
    return polys


def object_mask(img_path: str, shape, coco: Dict[str, List[np.ndarray]], grow_px: int) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    polys = list(coco.get(os.path.basename(img_path), []))
    txt = os.path.splitext(img_path)[0] + ".txt"
    if os.path.exists(txt):
        polys += _yolo_polys(txt, w, h)
    for p in polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    if grow_px > 0 and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1, 2 * grow_px + 1)))
    return m.astype(bool)


def build_plates(folders: List[str], out_dir: str, min_group: int = 5, similarity: float = 0.9,
                 grow_px: int = 20, max_inpaint_frac: float = 0.03, allow_single: bool = False,
                 log: Callable[[str], None] = print) -> List[str]:
    files = []
    coco: Dict[str, List[np.ndarray]] = {}
    for f in folders:
        files += sorted(p for p in glob.glob(os.path.join(glob.escape(f), "*")) if p.lower().endswith(IMG_EXT))
        coco.update(_coco_polys(f))
    if not files:
        raise RuntimeError("no images found")
    thumbs = []
    for p in files:
        im = _read(p)
        g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 40), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        thumbs.append((g - g.mean()) / (g.std() + 1e-6))
    T = np.stack(thumbs)
    left, groups = list(range(len(files))), []
    while left:
        c = T[left] @ T[left[0]] / T.shape[1]
        g = [left[k] for k in np.flatnonzero(c > similarity)]
        groups.append(g)
        left = [x for x in left if x not in g]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    for gi, g in enumerate(sorted(groups, key=len, reverse=True)):
        pairs = [(files[i], _read(files[i])) for i in g[:60]]
        shape = pairs[0][1].shape
        pairs = [(f, im) for f, im in pairs if im is not None and im.shape == shape]   # image and its mask together
        imgs = [im for _, im in pairs]
        masks = [object_mask(f, shape, coco, grow_px) for f, _ in pairs]
        if len(imgs) < min_group:
            # single views: only usable if the objects cover a small area (inpainting)
            # single views leave smudges / ghost shadows that look like objects -> opt-in only
            if allow_single and len(imgs) == 1 and masks[0].mean() <= max_inpaint_frac and masks[0].any():
                plate = cv2.inpaint(imgs[0], masks[0].astype(np.uint8), 7, cv2.INPAINT_TELEA)
                hole = 1.0
            else:
                continue
        else:
            st = np.stack(imgs).astype(np.float32)
            mk = np.stack(masks)[..., None]
            st[np.broadcast_to(mk, st.shape)] = np.nan
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                plate = np.nanmedian(st, axis=0)
            holes = np.isnan(plate[..., 0])
            plate = np.nan_to_num(plate).astype(np.uint8)
            hole = float(holes.mean())
            if hole > max_inpaint_frac:
                log(f"group {gi}: {len(imgs)} images, {hole * 100:.0f}% never free -> skipped")
                continue
            if holes.any():
                plate = cv2.inpaint(plate, holes.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        out = os.path.join(out_dir, f"plate_{gi:03d}_{Path(files[g[0]]).stem}.png")
        cv2.imencode(".png", plate)[1].tofile(out)
        written.append(out)
        log(f"group {gi}: {len(imgs)} image(s) -> {os.path.basename(out)}"
            + (f" (inpainted {hole * 100:.1f}%)" if hole and hole < 1 else " (inpainted labels)" if hole == 1.0 else ""))
    return written
