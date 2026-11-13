"""Dataset writers.

Native annotations (one JSON per camera image + one per multi-view frame) are the source of truth.
BOP is written while generating (it needs per-object masks); COCO / YOLO variants are exported from
the native annotations, so they can be re-exported with different filters without re-rendering:

    python -m syngt export <dataset_dir> --formats coco yolo_obb --min-visib 0.3
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from .config import ProjectConfig

SPLITS = ["train", "val", "test"]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or "cam"


def imwrite(path: Path, img: np.ndarray, params=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, params or [])
    if not ok:
        raise RuntimeError(f"encode failed for {path}")
    buf.tofile(str(path))


def write_label_png(path: Path, labels: np.ndarray, sixteen_bit: bool = False) -> None:
    """Write an id/label image. Indexed (palette) PNG - pixel value = id, but viewers show colours (Pascal VOC /
    DAVIS convention; read ids with PIL: np.array(Image.open(p))) - or, with sixteen_bit, a 16-bit grayscale PNG
    (read with cv2.IMREAD_UNCHANGED). The caller decides once per dataset, so all files have the same format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sixteen_bit and labels.max(initial=0) <= 255:
        from PIL import Image
        from .visualize import color_for
        im = Image.fromarray(labels.astype(np.uint8), mode="P")
        pal = [0, 0, 0]
        for i in range(1, 256):
            pal += list(color_for(i))
        im.putpalette(pal)
        im.save(str(path), optimize=False)
    else:
        imwrite(path, labels.astype(np.uint16))


def read_label_png(path: Path) -> np.ndarray:
    """Inverse of write_label_png (works for both palette and 16-bit files)."""
    from PIL import Image
    with Image.open(str(path)) as im:
        return np.array(im).astype(np.int32)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assign_split(scene_idx: int, ratios: List[float], seed: int) -> str:
    r = np.asarray((list(ratios) + [0, 0, 0])[:3], float)
    if r.sum() <= 0:
        return "train"
    r = r / r.sum()
    u = np.random.default_rng([seed, scene_idx, 777]).random()
    c = np.cumsum(r)
    for s, cc in zip(SPLITS, c):
        if u < cc:
            return s
    return "train"


def jdump(obj, path: Path, indent=None) -> None:
    """Write JSON atomically: a killed process never leaves a half-written file that breaks a later export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Per-scene writer (runs inside generation workers)
# --------------------------------------------------------------------------- #
class SceneWriter:
    def __init__(self, cfg: ProjectConfig, out_dir: Path, sampler):
        self.cfg, self.out, self.sampler = cfg, Path(out_dir), sampler
        self.multi_cam = len(cfg.cameras) > 1
        self.bop = "bop" in cfg.output.formats
        # one PNG format per dataset (never per image): 16 bit as soon as an id can exceed 255
        models = cfg.enabled_models()
        n_inst = (cfg.objects.total_count.max if cfg.objects.count_mode == "total"
                  else sum(max(m.count.min, m.count.max) for m in models))
        self.inst_16bit = n_inst > 255
        self.sem_16bit = max([m.class_id for m in models], default=0) > 255

    def stem(self, scene_idx: int, view_idx: int, cam_name: str) -> str:
        return f"{scene_idx:06d}_{view_idx:04d}_{safe_name(cam_name)}"

    def write_scene(self, st, frames_per_view: List[list], split: str) -> dict:
        o, out = self.cfg.output, self.out
        ext = "." + o.rgb_format
        stats = Counter()
        bop_gt: Dict[str, dict] = defaultdict(dict)
        bop_info: Dict[str, dict] = defaultdict(dict)
        bop_cam: Dict[str, dict] = defaultdict(dict)
        bop_scene_dir = out / "bop" / f"{split}_pbr" / f"{st.scene_idx:06d}"
        for view_idx, frames in enumerate(frames_per_view):
            group = {"scene_id": st.scene_idx, "view_id": view_idx, "split": split, "cameras": []}
            for fr in frames:
                stem = self.stem(st.scene_idx, view_idx, fr.cam_name)
                rgb_path = out / "rgb" / split / (stem + (".png" if fr.rgb.shape[2] == 4 else ext))
                bgr = cv2.cvtColor(fr.rgb, cv2.COLOR_RGBA2BGRA if fr.rgb.shape[2] == 4 else cv2.COLOR_RGB2BGR)
                params = [cv2.IMWRITE_JPEG_QUALITY, o.jpg_quality] if rgb_path.suffix == ".jpg" else []
                imwrite(rgb_path, bgr, params)
                files = {"rgb": rgb_path.relative_to(out).as_posix()}
                depth_u16 = None
                if o.save_depth and fr.depth is not None:
                    d = np.round(fr.depth * 1000.0 / o.depth_scale_mm)
                    d[(d < 0) | (d > 65535)] = 0     # out of the 16-bit range = "no measurement", never a wrong one
                    depth_u16 = d.astype(np.uint16)
                    p = out / "depth" / split / (stem + ".png")
                    imwrite(p, depth_u16)
                    files["depth"] = p.relative_to(out).as_posix()
                if o.save_instance_masks:
                    p = out / "masks" / split / (stem + ".png")
                    write_label_png(p, fr.inst, self.inst_16bit)
                    files["instance_mask"] = p.relative_to(out).as_posix()
                if o.save_semantic_masks:
                    p = out / "semantic" / split / (stem + ".png")
                    write_label_png(p, fr.semantic, self.sem_16bit)
                    files["semantic_mask"] = p.relative_to(out).as_posix()
                if o.save_normals and fr.normals is not None:
                    n = np.clip((fr.normals + 1) * 0.5 * 65535, 0, 65535).astype(np.uint16)
                    p = out / "normals" / split / (stem + ".png")
                    imwrite(p, cv2.cvtColor(n, cv2.COLOR_RGB2BGR))
                    files["normals"] = p.relative_to(out).as_posix()
                objs = []
                for a in fr.annotations:
                    a2 = {k: v for k, v in a.items() if not k.startswith("_")}
                    if o.save_amodal_masks and "_mask_full" in a:
                        p = out / "amodal" / split / f"{stem}_{a['inst_id']:03d}.png"
                        imwrite(p, a["_mask_full"].astype(np.uint8) * 255)
                        a2["amodal_mask"] = p.relative_to(out).as_posix()
                    objs.append(a2)
                    stats[f"class_{a['obj_id']}"] += 1
                stats["instances"] += len(objs)
                stats["images"] += 1
                cam = self.cfg.cameras[fr.cam_idx]
                ann = {
                    "image": {"stem": stem, "width": cam.width, "height": cam.height, "split": split,
                              "scene_id": st.scene_idx, "view_id": view_idx, "camera": fr.cam_name,
                              "camera_index": fr.cam_idx, "files": files, "background": fr.background_mode,
                              "has_alpha": fr.rgb.shape[2] == 4},
                    "camera": {"K": fr.K.tolist(), "dist": list(map(float, fr.dist)),
                               "T_world_cam": fr.T_world_cam.tolist(),
                               "T_cam_world": np.linalg.inv(fr.T_world_cam).tolist(),
                               "depth_scale_mm": o.depth_scale_mm, "depth_model": cam.depth.model,
                               "preset": cam.preset},
                    "objects": objs,
                    "sensor": fr.sensor_params,
                    "scene": st.info,
                }
                jdump(ann, out / "annotations" / split / f"{stem}.json")
                group["cameras"].append({"name": fr.cam_name, "stem": stem, "T_world_cam": fr.T_world_cam.tolist()})
                if self.bop:
                    self._bop_frame(bop_scene_dir, fr, view_idx, rgb_path, depth_u16, st, bop_gt, bop_info, bop_cam)
            jdump(group, out / "frames" / split / f"{st.scene_idx:06d}_{view_idx:04d}.json")
        if self.bop:
            for sfx in bop_gt:
                jdump(bop_gt[sfx], bop_scene_dir / f"scene_gt{sfx}.json")
                jdump(bop_info[sfx], bop_scene_dir / f"scene_gt_info{sfx}.json")
                jdump(bop_cam[sfx], bop_scene_dir / f"scene_camera{sfx}.json")
        return dict(stats)

    def _bop_frame(self, sdir: Path, fr, view_idx, rgb_path: Path, depth_u16, st, gt, info, camd):
        sfx = f"_{safe_name(fr.cam_name)}" if self.multi_cam else ""
        name = f"{view_idx:06d}"
        link_or_copy(rgb_path, sdir / f"rgb{sfx}" / (name + rgb_path.suffix))
        if depth_u16 is not None:
            imwrite(sdir / f"depth{sfx}" / (name + ".png"), depth_u16)
        bop_id = {a["inst_id"]: a["bop_obj_id"] for a in st.annotated}
        gts, infos = [], []
        for k, a in enumerate(fr.annotations):
            gts.append({"cam_R_m2c": a["cam_R_m2c"], "cam_t_m2c": a["cam_t_m2c_mm"], "obj_id": int(bop_id[a["inst_id"]])})
            bv = a["bbox_visib"] or [-1, -1, -1, -1]
            bo = [int(round(x)) for x in a["bbox_obj"]] if a["bbox_obj"] else [-1, -1, -1, -1]
            infos.append({"bbox_obj": bo, "bbox_visib": bv, "px_count_all": a["px_count_all"],
                          "px_count_valid": int(a["px_count_all"]), "px_count_visib": a["px_count_visib"],
                          "visib_fract": a["visib_fract"]})
            if "_mask_full" in a:
                imwrite(sdir / f"mask{sfx}" / f"{name}_{k:06d}.png", a["_mask_full"].astype(np.uint8) * 255)
                imwrite(sdir / f"mask_visib{sfx}" / f"{name}_{k:06d}.png", a["_mask_visib"].astype(np.uint8) * 255)
        gt[sfx][str(view_idx)] = gts
        info[sfx][str(view_idx)] = infos
        T_cw = np.linalg.inv(fr.T_world_cam)
        camd[sfx][str(view_idx)] = {"cam_K": fr.K.reshape(-1).tolist(), "depth_scale": self.cfg.output.depth_scale_mm,
                                    "cam_R_w2c": T_cw[:3, :3].reshape(-1).tolist(),
                                    "cam_t_w2c": (T_cw[:3, 3] * 1000).tolist()}


# --------------------------------------------------------------------------- #
# Exporters from native annotations
# --------------------------------------------------------------------------- #
def iter_annotations(out: Path, split: str) -> Iterable[dict]:
    """Annotations of all finished scenes. Scenes that were interrupted (cancel, crash) have no .done marker and
    are skipped - their files may be incomplete."""
    d = out / "annotations" / split
    if not d.exists():
        return
    prog = out / ".progress"
    done = {p.stem for p in prog.glob("*.done")} if prog.is_dir() else None
    for p in sorted(d.glob("*.json")):
        if done is not None and p.name.split("_")[0] not in done:
            continue
        with open(p, "r", encoding="utf-8") as f:
            yield json.load(f)


def keep_for_detection(a: dict, o) -> bool:
    return (a.get("px_count_visib", 0) >= o.min_px_visib and a.get("visib_fract", 0) >= o.min_visib_fract
            and a.get("truncation", 0) <= o.max_truncation and a.get("bbox_visib"))


def _obb(a: dict, src: str) -> Optional[dict]:
    key = {"visible_mask": "obb_visib", "amodal_mask": "obb_amodal", "projected_3d_box": "obb_box3d"}[src]
    return a.get(key) or a.get("obb_visib")


def _obb_in_image(corners: np.ndarray, W: int, H: int) -> Optional[np.ndarray]:
    """Oriented box (4x2 px) for a label that must lie inside the image. A box that sticks out is cut at the
    image border and re-fitted (clipping the corners one by one would turn it into a skewed quad that no longer
    covers the object)."""
    c = np.asarray(corners, np.float32).reshape(4, 2)
    if c[:, 0].min() >= 0 and c[:, 1].min() >= 0 and c[:, 0].max() <= W and c[:, 1].max() <= H:
        return c.astype(float)
    rect = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    area, inter = cv2.intersectConvexConvex(cv2.convexHull(c), rect)
    if area < 1.0 or inter is None or len(inter) < 3:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(inter.reshape(-1, 2)))
    if box[:, 0].min() < -0.5 or box[:, 1].min() < -0.5 or box[:, 0].max() > W + 0.5 or box[:, 1].max() > H + 0.5:
        x0, y0 = inter.reshape(-1, 2).min(0)        # the re-fitted box still pokes out: axis-aligned always fits
        x1, y1 = inter.reshape(-1, 2).max(0)
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    return np.clip(box.astype(float), [0, 0], [W, H])


def class_table(cfg: ProjectConfig) -> Dict[int, str]:
    names = {}
    for m in cfg.objects.models:
        if m.enabled and m.path:
            names.setdefault(int(m.class_id), m.class_name)
    return dict(sorted(names.items()))


def export_all(cfg: ProjectConfig, out: Path, formats: Optional[List[str]] = None, log=print) -> dict:
    out = Path(out)
    formats = formats or cfg.output.formats
    classes = class_table(cfg)
    summary = {}
    if "coco" in formats:
        summary["coco"] = export_coco(cfg, out, classes)
        log("COCO export done")
    if "coco_bb" in formats:
        summary["coco_bb"] = export_coco_bb(cfg, out, classes)
        log(f"COCO boxes (train/ + validation/ folders) export done -> {out / 'coco_bb'}")
    for f in ("yolo", "yolo_obb", "yolo_seg", "yolo_pose"):
        if f in formats:
            summary[f] = export_yolo(cfg, out, classes, f)
            log(f"{f} export done")
    return summary


def export_coco(cfg: ProjectConfig, out: Path, classes: Dict[int, str]) -> dict:
    o = cfg.output
    nkp = 9 if o.keypoints == "bbox3d_9" else 8
    kp_names = (["center"] if nkp == 9 else []) + [f"corner_{i}" for i in range(8)]
    cats = [{"id": cid, "name": n, "supercategory": "object", "keypoints": kp_names,
             "skeleton": [[a + (nkp - 8) + 1, b + (nkp - 8) + 1] for a, b in
                          ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))]}
            for cid, n in classes.items()]
    counts = {}
    for split in SPLITS:
        images, anns = [], []
        for img_id, A in enumerate(iter_annotations(out, split), start=1):
            im = A["image"]
            images.append({"id": img_id, "file_name": im["files"]["rgb"], "width": im["width"], "height": im["height"],
                           "scene_id": im["scene_id"], "view_id": im["view_id"], "camera": im["camera"],
                           "depth_file": im["files"].get("depth"), "cam_K": A["camera"]["K"],
                           "dist": A["camera"]["dist"], "T_world_cam": A["camera"]["T_world_cam"]})
            for a in A["objects"]:
                if not keep_for_detection(a, o):
                    continue
                kps = []
                for u, v, vis in a["keypoints"]:
                    kps += [round(u, 2), round(v, 2), int(vis)] if vis > 0 else [0, 0, 0]
                # polygons cannot describe holes (bores, rings): use the exact RLE mask for those
                seg = (a.get("rle_visib") if a.get("seg_has_holes") else None) or a["segmentation_polygons"] \
                    or a.get("rle_visib")
                anns.append({
                    "id": len(anns) + 1, "image_id": img_id, "category_id": a["obj_id"],
                    "bbox": [float(x) for x in a["bbox_visib"]], "area": float(a["px_count_visib"]),
                    "iscrowd": 0, "segmentation": seg,
                    "bbox_amodal": a["bbox_obj"], "visib_fract": a["visib_fract"], "truncation": a["truncation"],
                    "obb": _obb(a, o.obb_source),
                    "keypoints": kps, "num_keypoints": int(sum(1 for i in range(2, len(kps), 3) if kps[i] > 0)),
                    "pose": {"R": a["cam_R_m2c"], "t_mm": a["cam_t_m2c_mm"]}, "inst_id": a["inst_id"],
                    "rle_amodal": a.get("rle_amodal"),
                })
        if images:
            jdump({"info": {"description": cfg.name, "generator": "syngt"}, "images": images, "annotations": anns,
                   "categories": cats}, out / "coco" / f"{split}.json")
            counts[split] = {"images": len(images), "annotations": len(anns)}
    return counts


COCO_BB_SPLITS = {"train": "train", "val": "validation", "test": "test"}    # syngt split -> folder / file suffix


def export_coco_bb(cfg: ProjectConfig, out: Path, classes: Dict[int, str]) -> dict:
    """Plain COCO bounding boxes, one self-contained folder per split:

        coco_bb/train/coco_train.json            + the training images next to it
        coco_bb/validation/coco_validation.json  + the validation images next to it

    This is the layout read by `supervision.DetectionDataset.from_coco` (RT-DETR trainers, Roboflow-style
    tools): `file_name` is the bare image name inside the split folder, category ids are 1..N in class order."""
    import datetime
    o = cfg.output
    root = out / "coco_bb"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    cat_id = {cid: i for i, cid in enumerate(sorted(classes), start=1)}
    cats = [{"id": cat_id[cid], "name": classes[cid], "supercategory": ""} for cid in sorted(classes)]
    counts = {}
    for split in SPLITS:
        folder = root / COCO_BB_SPLITS[split]
        images, anns = [], []
        for img_id, A in enumerate(iter_annotations(out, split), start=1):
            im = A["image"]
            W, H = im["width"], im["height"]
            src = out / im["files"]["rgb"]
            link_or_copy(src, folder / src.name)
            images.append({"id": img_id, "file_name": src.name, "width": W, "height": H})
            for a in A["objects"]:
                if not keep_for_detection(a, o) or a["obj_id"] not in cat_id:
                    continue
                x, y, w, h = a["bbox_visib"]
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue        # degenerate boxes break the trainers' augmentation
                box = [round(float(x0), 2), round(float(y0), 2), round(float(x1 - x0), 2), round(float(y1 - y0), 2)]
                anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": cat_id[a["obj_id"]],
                             "iscrowd": 0, "bbox": box, "area": round(box[2] * box[3], 2)})
        if images:
            jdump({"info": {"description": f"{cfg.name} (generated by syngt)", "version": "1.0",
                            "year": str(datetime.date.today().year), "contributor": "",
                            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                   "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
                   "categories": cats, "images": images, "annotations": anns},
                  folder / f"coco_{COCO_BB_SPLITS[split]}.json")
            counts[split] = {"images": len(images), "annotations": len(anns), "folder": str(folder)}
    return counts


def export_yolo(cfg: ProjectConfig, out: Path, classes: Dict[int, str], variant: str) -> dict:
    o = cfg.output
    root = out / variant
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    cls_map = {cid: i for i, cid in enumerate(sorted(classes))}  # dense 0..N-1
    counts = {}
    for split in SPLITS:
        n_img = n_lab = 0
        for A in iter_annotations(out, split):
            im = A["image"]
            W, H = im["width"], im["height"]
            src = out / im["files"]["rgb"]
            dst_img = root / "images" / split / src.name
            link_or_copy(src, dst_img)
            lines = []
            for a in A["objects"]:
                if not keep_for_detection(a, o) or a["obj_id"] not in cls_map:
                    continue
                c = cls_map[a["obj_id"]]
                x, y, w, h = a["bbox_visib"]
                x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                bb = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
                if variant == "yolo":
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in bb))
                elif variant == "yolo_obb":
                    ob = _obb(a, o.obb_source)
                    if not ob:
                        continue
                    pts = _obb_in_image(np.asarray(ob["corners"], float), W, H)
                    if pts is None:
                        continue
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in (pts / [W, H]).reshape(-1)))
                elif variant == "yolo_seg":
                    polys = a["segmentation_polygons"]
                    if not polys:
                        continue
                    flat = polys[0]
                    if (len(polys) > 1 or a.get("seg_has_holes")) and a.get("rle_visib"):
                        # YOLO allows one polygon per object: join pieces and holes instead of dropping them
                        from .annotations import rle_decode, single_polygon
                        flat = single_polygon(rle_decode(a["rle_visib"]), o.polygon_tolerance_px) or flat
                    elif len(polys) > 1:
                        flat = max(polys, key=len)
                    poly = np.asarray(flat, float).reshape(-1, 2)
                    poly[:, 0] /= W
                    poly[:, 1] /= H
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in np.clip(poly, 0, 1).reshape(-1)))
                elif variant == "yolo_pose":
                    kp = []
                    for u, v, vis in a["keypoints"]:
                        if vis == 0:
                            kp += [0.0, 0.0, 0]
                        else:
                            kp += [u / W, v / H, int(vis)]
                    lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in bb) + " " +
                                 " ".join(f"{v:.6f}" if i % 3 != 2 else str(int(v)) for i, v in enumerate(kp)))
            lp = root / "labels" / split / (src.stem + ".txt")
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_img += 1
            n_lab += len(lines)
        if n_img:
            counts[split] = {"images": n_img, "labels": n_lab}
    names = {cls_map[cid]: classes[cid] for cid in sorted(classes)}
    y = [f"# Generated by syngt ({variant}). class index = dense index of sorted class_id",
         f"path: {root.resolve().as_posix()}"]
    for split in SPLITS:
        if split in counts:
            y.append(f"{split}: images/{split}")
    if variant == "yolo_pose":
        nk = 9 if o.keypoints == "bbox3d_9" else 8
        y.append(f"kpt_shape: [{nk}, 3]")
        y.append(f"flip_idx: {list(range(nk))}  # 3D-box keypoints are not mirror-symmetric: train with fliplr=0.0")
    y.append("names:")
    for i, n in names.items():
        y.append(f"  {i}: {n}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("\n".join(y) + "\n", encoding="utf-8")     # never the locale's encoding
    # one class name per line, in label-index order (read by many training / labelling tools)
    (root / "classes.txt").write_text("\n".join(names[i] for i in sorted(names)) + "\n", encoding="utf-8")
    jdump({"class_id_to_yolo": cls_map, "names": {str(i): n for i, n in names.items()}}, root / "class_map.json",
          indent=2)
    return counts


# everything a generation run writes into a dataset folder (and nothing else is ever deleted from it)
DATASET_ENTRIES = ("rgb", "depth", "masks", "semantic", "normals", "amodal", "annotations", "frames", "bop", "coco",
                   "coco_bb",
                   "yolo", "yolo_obb", "yolo_seg", "yolo_pose", ".progress", "cameras.json", "dataset_info.json",
                   "config.yaml")


def clear_dataset(out: Path) -> int:
    """Remove a previous syngt dataset from `out` (start fresh). Only files/folders that syngt itself writes are
    touched, and only if the folder is a syngt dataset. Returns the number of removed entries."""
    import shutil
    out = Path(out)
    if not ((out / "config.yaml").exists() or (out / ".progress").is_dir()):
        return 0
    n = 0
    for name in DATASET_ENTRIES:
        p = out / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        elif p.exists() or p.is_symlink():
            p.unlink()
            n += 1
    return n


def write_static(cfg: ProjectConfig, out: Path, sampler) -> None:
    """Things that do not depend on scenes: config copy, cameras/rig, BOP models + models_info."""
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")
    from .viewpoints import rig_T
    cams = []
    for c in cfg.cameras:
        cams.append({"name": c.name, "preset": c.preset, "width": c.width, "height": c.height,
                     "K": c.K().tolist(), "dist": c.dist, "T_rig_cam": rig_T(c).tolist(),
                     "world_pose": c.world_pose, "depth": c.depth.model_dump()})
    jdump({"cameras": cams, "rig_mode": cfg.viewpoints.rig_mode,
           "conventions": {"camera": "OpenCV (x right, y down, z forward)", "units": "metres (BOP files: mm)",
                           "T_a_b": "maps points from frame b to frame a", "world": "z up, support plane z=0"}},
          out / "cameras.json", indent=2)
    if "bop" in cfg.output.formats:
        from .assets import export_bop_model, symmetry_info
        info = {}
        for mi, (m, a) in enumerate(zip(sampler.models, sampler.assets)):
            oid = sampler.bop_obj_id(mi)
            e = export_bop_model(a, out / "bop" / "models" / f"obj_{oid:06d}.ply")
            e.update(symmetry_info(m.symmetry, m.symmetry_order, a))
            info[str(oid)] = e
        jdump(info, out / "bop" / "models" / "models_info.json", indent=2)
        jdump({str(sampler.bop_obj_id(i)): {"name": m.class_name, "class_id": m.class_id,
                                            "source": m.path} for i, m in enumerate(sampler.models)},
              out / "bop" / "obj_id_map.json", indent=2)
        c0 = cfg.cameras[0]
        jdump({"cx": c0.cx, "cy": c0.cy, "fx": c0.fx, "fy": c0.fy, "width": c0.width, "height": c0.height,
               "depth_scale": cfg.output.depth_scale_mm}, out / "bop" / "camera.json", indent=2)


def dataset_stats(out: Path) -> dict:
    st = {"splits": {}, "per_class": Counter(), "visib_hist": [0] * 10}
    for split in SPLITS:
        n = 0
        for A in iter_annotations(out, split):
            n += 1
            for a in A["objects"]:
                st["per_class"][a["class_name"]] += 1
                st["visib_hist"][min(9, int(a["visib_fract"] * 10))] += 1
        if n:
            st["splits"][split] = n
    st["per_class"] = dict(st["per_class"])
    return st
