"""Per-camera, per-instance ground truth computation."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import inv_T, project, transform_points
from .imaging import CameraCanvas


def mask_bbox(m: np.ndarray, off=(0, 0)):
    cols = np.flatnonzero(m.any(0))
    if len(cols) == 0:
        return None
    rows = np.flatnonzero(m.any(1))
    return [int(cols[0] + off[0]), int(rows[0] + off[1]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]


def rle_encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    m = mask if (mask.dtype == np.uint8 and mask.flags.f_contiguous) else np.asfortranarray(mask, dtype=np.uint8)
    r = mu.encode(m)
    r["counts"] = r["counts"].decode("ascii")
    return r


def rle_encode_window(mc: np.ndarray, x0: int, y0: int, H: int, W: int) -> dict:
    """COCO RLE of a full HxW mask that is only non-zero in the window mc at (x0, y0) - avoids
    building / copying full-frame C-ordered masks."""
    mt = np.zeros((W, H), np.uint8)                      # transposed == Fortran layout of (H, W)
    mt[x0:x0 + mc.shape[1], y0:y0 + mc.shape[0]] = mc.T
    return rle_encode(mt.T)


def full_frame(mc: np.ndarray, x0: int, y0: int, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y0 + mc.shape[0], x0:x0 + mc.shape[1]] = mc
    return m


def rle_decode(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("ascii")
    return mu.decode(r).astype(bool)


def mask_has_holes(mask: np.ndarray, min_frac: float = 0.01, min_area: float = 4.0) -> bool:
    """True if the mask has holes that matter (a bore, the inside of a ring): polygons alone cannot describe it."""
    m = mask.astype(np.uint8)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    hole = sum(cv2.contourArea(c) for c, h in zip(cs, hier[0]) if h[3] >= 0)
    return hole >= max(min_area, min_frac * float(m.sum()))


def single_polygon(mask: np.ndarray, tol: float = 1.0, min_area: float = 4.0, off=(0, 0)) -> List[float]:
    """The mask as ONE polygon (flat x,y list) for formats that allow only one per object (YOLO-seg).
    Separate pieces and holes are joined to the largest outline by zero-width bridges, so filling the polygon
    reproduces the mask (holes stay holes) instead of dropping pieces or filling bores."""
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE, offset=tuple(off))
    if hier is None:
        return []
    rings = []
    for c, h in zip(cs, hier[0]):
        if cv2.contourArea(c) < min_area:
            continue
        if tol > 0:
            c = cv2.approxPolyDP(c, tol, True)
        if len(c) >= 3:
            rings.append(c.reshape(-1, 2).astype(np.float64))
    if not rings:
        return []
    rings.sort(key=lambda r: -abs(cv2.contourArea(r.astype(np.float32))))
    poly = rings[0]
    for r in rings[1:]:     # attach at the closest pair of vertices: ... P[i], R[j..], R[..j], P[i], ...
        d = np.linalg.norm(poly[:, None, :] - r[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        poly = np.vstack([poly[:i + 1], r[j:], r[:j + 1], poly[i:]])
    return [float(v) for v in poly.reshape(-1)]


def mask_polygons(mask: np.ndarray, tol: float = 1.0, min_area: float = 4.0, off=(0, 0)) -> List[List[float]]:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=tuple(off))
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        if tol > 0:
            c = cv2.approxPolyDP(c, tol, True)
        if len(c) >= 3:
            out.append([float(v) for v in c.reshape(-1)])
    return out


def obb_from_points(pts: np.ndarray) -> Optional[dict]:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    corners = cv2.boxPoints(((cx, cy), (w, h), ang))
    return {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h), "angle_deg": float(ang),
            "corners": [[float(x), float(y)] for x, y in corners]}


def _mask_contour_points(m: np.ndarray, off=(0, 0)) -> np.ndarray:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=tuple(off))
    if not cs:
        return np.zeros((0, 2), np.float32)
    return np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32)


def _extent_beyond_canvas(asset, scale: float, T_cam_obj: np.ndarray, canvas: CameraCanvas, px_in_canvas: int):
    """For an object that leaves the render canvas: (estimated full silhouette area in px, hull outline in canvas
    px, fraction of the hull behind the camera). The projected convex hull bounds the silhouette; the part of it
    inside the canvas tells how much of the hull the real silhouette fills. None if nothing is in front."""
    hv = asset.hull_vertices if asset.hull_vertices is not None else asset.vertices
    pc = transform_points(T_cam_obj, np.asarray(hv, np.float64) * scale)
    front = pc[:, 2] > 1e-3
    if front.sum() < 3:
        return None
    uv = pc[front, :2] / pc[front, 2:3] * [canvas.Kr[0, 0], canvas.Kr[1, 1]] + [canvas.Kr[0, 2], canvas.Kr[1, 2]]
    uv = np.clip(uv, -1e5, 1e5).astype(np.float32)
    hull = cv2.convexHull(uv)
    area_full = float(cv2.contourArea(hull))
    rect = np.array([[0, 0], [canvas.Wr, 0], [canvas.Wr, canvas.Hr], [0, canvas.Hr]], np.float32)
    area_in, _ = cv2.intersectConvexConvex(hull, rect)
    if area_in <= 1.0 or area_full <= 1.0:
        return None
    fill = float(np.clip(px_in_canvas / area_in, 0.05, 1.0))
    return max(float(px_in_canvas), area_full * fill), hull.reshape(-1, 2).astype(np.float64), float(1 - front.mean())


def compute_annotations(instances: List[dict], canvas: CameraCanvas, inst_canvas: np.ndarray,
                        solo: Dict[int, tuple], inst_img: np.ndarray, depth_img: np.ndarray,
                        T_world_cam: np.ndarray, K: np.ndarray, dist, polygon_tol: float = 1.0,
                        keypoint_mode: str = "bbox3d_9", want_rle: bool = True,
                        want_full_masks: bool = False) -> List[dict]:
    """``instances``: dicts with inst_id, obj_id, class_name, model_name, asset, T_world, scale."""
    H, W = inst_img.shape
    region = canvas.image_region_in_canvas()
    T_cam_world = inv_T(T_world_cam)
    out = []
    for it in instances:
        iid = it["inst_id"]
        asset = it["asset"]
        T_cam_obj = T_cam_world @ it["T_world"]
        scale = it.get("scale", 1.0)
        crop, (x0, y0) = solo.get(iid, (np.zeros((0, 0), bool), (0, 0)))
        total_canvas = int(crop.sum())
        ch, cw = crop.shape
        # ---- image-space window holding the object's (amodal) silhouette; all mask work happens inside it
        if not canvas.distorted and total_canvas:
            ix0, iy0 = x0 - canvas.px, y0 - canvas.py
            wx0, wy0 = max(0, ix0), max(0, iy0)
            wx1, wy1 = min(W, ix0 + cw), min(H, iy0 + ch)
            if wx1 > wx0 and wy1 > wy0:
                full_c = crop[wy0 - iy0:wy1 - iy0, wx0 - ix0:wx1 - ix0].copy()
            else:
                wx0 = wy0 = 0
                wx1 = wy1 = 0
                full_c = np.zeros((0, 0), bool)
        else:
            if canvas.distorted and total_canvas:
                fc = np.zeros(inst_canvas.shape, bool)
                fc[y0:y0 + ch, x0:x0 + cw] = crop
                full = canvas.to_image(fc)
            else:
                full = inst_img == iid
            bb = mask_bbox(full)
            if bb is None:
                wx0 = wy0 = wx1 = wy1 = 0
            else:
                wx0, wy0, wx1, wy1 = bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]
            full_c = full[wy0:wy1, wx0:wx1]
        visib_c = inst_img[wy0:wy1, wx0:wx1] == iid
        full_c = full_c | visib_c  # numerical safety (AA-free buffers already agree)
        px_all, px_vis = int(full_c.sum()), int(visib_c.sum())
        if total_canvas == 0 and px_vis == 0:
            continue  # outside the view entirely
        in_region = int((crop & region[y0:y0 + ch, x0:x0 + cw]).sum()) if total_canvas else 0
        truncation = 1.0 - in_region / total_canvas if total_canvas else 1.0
        off = (wx0, wy0)
        # the silhouette is only known inside the render canvas: if it reaches the canvas border, part of the
        # object lies beyond it -> estimate the full extent from the projected convex hull
        beyond = None
        if total_canvas and (x0 <= 0 or y0 <= 0 or x0 + cw >= canvas.Wr or y0 + ch >= canvas.Hr):
            beyond = _extent_beyond_canvas(asset, scale, T_cam_obj, canvas, total_canvas)
            if beyond is not None:
                truncation = max(truncation, 1.0 - in_region / beyond[0], beyond[2])
        # amodal bbox (can exceed the image)
        if total_canvas:
            cpts = _mask_contour_points(crop, (x0, y0))
            if beyond is not None and not canvas.distorted:
                cpts = np.vstack([cpts, beyond[1]])
            ipts = canvas.canvas_to_image_points(cpts)
            bx0, by0 = ipts.min(0)
            bx1, by1 = ipts.max(0)
            bbox_obj = [float(bx0), float(by0), float(bx1 - bx0 + 1), float(by1 - by0 + 1)]
            amodal_pts = ipts
        else:
            bbox_obj = mask_bbox(full_c, off)
            amodal_pts = _mask_contour_points(full_c, off)
        bbox_vis = mask_bbox(visib_c, off)
        # 3D box keypoints
        kp_obj = asset.bbox3d_keypoints() * scale
        if keypoint_mode == "bbox3d_8":
            kp_obj = kp_obj[1:]
        kp_cam = transform_points(T_cam_obj, kp_obj)
        kp_uv = project(K, kp_cam, dist)
        kps = []
        for j, ((u, v), z) in enumerate(zip(kp_uv, kp_cam[:, 2])):
            inside = z > 0 and 0 <= u < W and 0 <= v < H
            if not inside:
                kps.append([float(u), float(v), 0])
                continue
            if keypoint_mode == "bbox3d_9" and j == 0:
                kps.append([float(u), float(v), 2 if px_vis > 0 else 1])
                continue
            dz = depth_img[int(v), int(u)]
            tol = max(0.004, 0.03 * asset.diameter * scale)
            occluded = dz > 0 and dz < z - tol
            kps.append([float(u), float(v), 1 if occluded else 2])
        corners_uv = project(K, transform_points(T_cam_obj, asset.bbox_corners() * scale), dist)
        origin_uv = project(K, T_cam_obj[:3, 3][None], dist)[0]
        ann = {
            "inst_id": iid,
            "obj_id": it["obj_id"],
            "class_name": it["class_name"],
            "model": it["model_name"],
            "scale": scale,
            "T_cam_obj": T_cam_obj.tolist(),
            "T_world_obj": it["T_world"].tolist(),
            "cam_R_m2c": T_cam_obj[:3, :3].reshape(-1).tolist(),
            "cam_t_m2c_mm": (T_cam_obj[:3, 3] * 1000.0).tolist(),
            "bbox_obj": bbox_obj,
            "bbox_visib": bbox_vis,
            "px_count_all": px_all,
            "px_count_visib": px_vis,
            "visib_fract": float(px_vis / px_all) if px_all else 0.0,
            "truncation": float(np.clip(truncation, 0, 1)),
            "obb_visib": obb_from_points(_mask_contour_points(visib_c, off)) if px_vis >= 3 else None,
            "obb_amodal": obb_from_points(amodal_pts) if len(amodal_pts) >= 3 else None,
            "obb_box3d": obb_from_points(corners_uv)
            if np.all(transform_points(T_cam_obj, asset.bbox_corners() * scale)[:, 2] > 0) else None,
            "keypoints": kps,
            "bbox3d_corners_px": corners_uv.tolist(),
            "origin_px": origin_uv.tolist(),
            "segmentation_polygons": mask_polygons(visib_c, polygon_tol, off=off) if px_vis else [],
            "seg_has_holes": bool(px_vis and mask_has_holes(visib_c)),
        }
        if want_rle:
            ann["rle_visib"] = rle_encode_window(visib_c, wx0, wy0, H, W) if px_vis else None
            ann["rle_amodal"] = rle_encode_window(full_c, wx0, wy0, H, W) if px_all else None
        if want_full_masks:
            ann["_mask_full"] = full_frame(full_c, wx0, wy0, H, W)
            ann["_mask_visib"] = full_frame(visib_c, wx0, wy0, H, W)
        out.append(ann)
    return out
