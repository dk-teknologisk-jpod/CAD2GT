"""Camera presets (Intel RealSense / Orbbec) and calibration-file import.

Preset intrinsics are *nominal* values derived from datasheet fields of view
(square pixels, centred principal point, no distortion). For best sim2real,
import the real calibration of your unit (rs-enumerate-devices -c JSON,
OpenCV YAML/XML, ROS camera_info, Kalibr camchain, or a plain JSON).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from .config import CameraConfig, DepthConfig


@dataclass
class StreamSpec:
    native: Tuple[int, int]
    hfov: float
    vfov: float
    resolutions: List[Tuple[int, int]]


@dataclass
class CameraPreset:
    vendor: str
    model: str
    color: StreamSpec
    depth: StreamSpec
    depth_type: str            # stereo | tof | structured_light
    baseline_m: float
    min_z: float
    max_z: float
    notes: str = ""
    extra_depth_modes: Dict[str, StreamSpec] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.vendor} {self.model}"


_RS_COLOR_169 = [(1920, 1080), (1280, 720), (960, 540), (848, 480), (640, 480), (640, 360), (424, 240), (320, 240)]
_RS_DEPTH = [(1280, 720), (848, 480), (640, 480), (640, 360), (480, 270), (424, 240)]
_RS_COLOR_D455 = [(1280, 800), (1280, 720), (848, 480), (640, 480), (640, 400), (640, 360), (424, 240)]
_OB_G2_COLOR = [(1920, 1080), (1280, 720), (640, 480), (640, 360)]
_OB_G2_DEPTH = [(1280, 800), (640, 400), (320, 200)]
_OB_335_COLOR = [(1280, 800), (1280, 720), (848, 480), (640, 480), (640, 400), (640, 360), (480, 270), (424, 240)]
_OB_335_DEPTH = [(1280, 800), (1280, 720), (848, 480), (640, 480), (640, 400), (640, 360), (480, 270), (424, 240)]
_FEMTO_COLOR = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720)]


def S(native, hfov, vfov, res) -> StreamSpec:
    return StreamSpec(tuple(native), hfov, vfov, list(res))


PRESETS: List[CameraPreset] = [
    # ---------------------------------------------------------------- Intel RealSense
    CameraPreset("Intel RealSense", "D405", S((1280, 720), 87, 58, _RS_DEPTH), S((1280, 720), 87, 58, _RS_DEPTH),
                 "stereo", 0.018, 0.07, 0.5, "Short-range; colour from the left imager (aligned by design)."),
    CameraPreset("Intel RealSense", "D415", S((1920, 1080), 69, 42, _RS_COLOR_169), S((1280, 720), 65, 40, _RS_DEPTH),
                 "stereo", 0.055, 0.16, 10.0, "Rolling-shutter depth imagers, narrow FOV."),
    CameraPreset("Intel RealSense", "D435 / D435i", S((1920, 1080), 69, 42, _RS_COLOR_169), S((1280, 720), 87, 58, _RS_DEPTH),
                 "stereo", 0.050, 0.28, 10.0, "Global-shutter depth imagers, wide FOV."),
    CameraPreset("Intel RealSense", "D455 / D457", S((1280, 800), 90, 65, _RS_COLOR_D455), S((1280, 720), 87, 58, _RS_DEPTH),
                 "stereo", 0.095, 0.4, 20.0, "Global-shutter RGB, long baseline."),
    CameraPreset("Intel RealSense", "D555", S((1280, 800), 90, 65, _RS_COLOR_D455), S((1280, 720), 87, 58, _RS_DEPTH),
                 "stereo", 0.095, 0.4, 20.0, "PoE; values assumed close to D455."),
    CameraPreset("Intel RealSense", "L515", S((1920, 1080), 70, 43, _RS_COLOR_169),
                 S((1024, 768), 70, 55, [(1024, 768), (640, 480), (320, 240)]),
                 "tof", 0.0, 0.25, 9.0, "LiDAR (discontinued)."),
    # ---------------------------------------------------------------- Orbbec
    CameraPreset("Orbbec", "Gemini 2", S((1920, 1080), 86, 55, _OB_G2_COLOR), S((1280, 800), 91, 66, _OB_G2_DEPTH),
                 "stereo", 0.050, 0.15, 10.0),
    CameraPreset("Orbbec", "Gemini 2 L", S((1920, 1080), 86, 55, _OB_G2_COLOR), S((1280, 800), 94, 68, _OB_G2_DEPTH),
                 "stereo", 0.100, 0.25, 10.0),
    CameraPreset("Orbbec", "Gemini 2 XL", S((1920, 1080), 86, 55, _OB_G2_COLOR), S((1280, 800), 91, 66, _OB_G2_DEPTH),
                 "stereo", 0.150, 0.4, 20.0, "Long baseline; nominal FOV assumed close to Gemini 2."),
    CameraPreset("Orbbec", "Gemini 335 / 336", S((1280, 800), 86, 55, _OB_335_COLOR), S((1280, 800), 90, 65, _OB_335_DEPTH),
                 "stereo", 0.050, 0.10, 10.0),
    CameraPreset("Orbbec", "Gemini 335L / 336L", S((1280, 800), 94, 68, _OB_335_COLOR), S((1280, 800), 94, 68, _OB_335_DEPTH),
                 "stereo", 0.095, 0.17, 20.0),
    CameraPreset("Orbbec", "Gemini 335Le", S((1280, 800), 94, 68, _OB_335_COLOR), S((1280, 800), 94, 68, _OB_335_DEPTH),
                 "stereo", 0.095, 0.17, 20.0, "GigE variant."),
    CameraPreset("Orbbec", "Astra 2", S((1920, 1080), 86, 55, [(1920, 1080), (1280, 720), (640, 480)]),
                 S((1600, 1200), 75, 58, [(1600, 1200), (800, 600), (400, 300)]),
                 "structured_light", 0.075, 0.6, 8.0),
    CameraPreset("Orbbec", "Femto Bolt / Mega (NFOV)", S((3840, 2160), 80, 51, _FEMTO_COLOR),
                 S((640, 576), 75, 65, [(640, 576), (320, 288)]), "tof", 0.0, 0.25, 5.46,
                 "iToF, Azure-Kinect compatible."),
    CameraPreset("Orbbec", "Femto Bolt / Mega (WFOV)", S((3840, 2160), 80, 51, _FEMTO_COLOR),
                 S((1024, 1024), 120, 120, [(1024, 1024), (512, 512)]), "tof", 0.0, 0.25, 2.88,
                 "WFOV depth has strong fisheye distortion in reality - import calibration for exact use."),
    CameraPreset("Microsoft", "Azure Kinect (NFOV)", S((3840, 2160), 90, 59, _FEMTO_COLOR),
                 S((640, 576), 75, 65, [(640, 576), (320, 288)]), "tof", 0.0, 0.5, 3.86),
    # ---------------------------------------------------------------- generic
    CameraPreset("Generic", "Industrial 5MP (2/3\", 12 mm)", S((2448, 2048), 39, 33, [(2448, 2048), (1224, 1024)]),
                 S((2448, 2048), 39, 33, [(2448, 2048)]), "ideal", 0.0, 0.1, 10.0),
    CameraPreset("Generic", "Industrial 12MP (1.1\", 16 mm)", S((4096, 3000), 46, 34, [(4096, 3000), (2048, 1500)]),
                 S((4096, 3000), 46, 34, [(4096, 3000)]), "ideal", 0.0, 0.1, 10.0),
    CameraPreset("Generic", "Webcam 1080p 78 deg", S((1920, 1080), 70.4, 43.3, [(1920, 1080), (1280, 720), (640, 480)]),
                 S((1920, 1080), 70.4, 43.3, [(1920, 1080)]), "ideal", 0.0, 0.1, 10.0),
]

PRESET_BY_KEY: Dict[str, CameraPreset] = {p.key: p for p in PRESETS}


def intrinsics_for(spec: StreamSpec, width: int, height: int) -> Tuple[float, float, float, float]:
    """Nominal intrinsics for a stream mode. Lower modes are scaled (and centre-cropped
    when the aspect ratio differs) from the native sensor mode, like the vendors do."""
    W, H = spec.native
    fx_n = (W / 2.0) / math.tan(math.radians(spec.hfov) / 2.0)
    fy_n = (H / 2.0) / math.tan(math.radians(spec.vfov) / 2.0)
    # datasheets round FOVs; physical pixels are ~square -> average
    f_n = 0.5 * (fx_n + fy_n) if abs(fx_n - fy_n) / fx_n < 0.08 else fx_n
    fx_n = fy_n = f_n
    s = max(width / W, height / H)
    return fx_n * s, fy_n * s, (width - 1) / 2.0, (height - 1) / 2.0


def preset_slug(key: str) -> str:
    """Short camera name from a preset, e.g. 'Orbbec Gemini 335 / 336' -> 'gemini335'."""
    import re
    model = PRESET_BY_KEY[key].model if key in PRESET_BY_KEY else key
    model = model.split("/")[0].split("(")[0]
    return re.sub(r"[^a-z0-9]+", "", model.lower()) or "cam"


def is_auto_camera_name(name: str) -> bool:
    import re
    slugs = {preset_slug(k) for k in PRESET_BY_KEY}
    return bool(re.fullmatch(r"cam\d*", name)) or re.sub(r"_\d+$", "", name) in slugs


def camera_from_preset(key: str, stream: str = "color", width: Optional[int] = None,
                       height: Optional[int] = None, name: str = "cam0") -> CameraConfig:
    p = PRESET_BY_KEY[key]
    spec = p.color if stream == "color" else p.depth
    w, h = (width, height) if width and height else spec.native
    fx, fy, cx, cy = intrinsics_for(spec, w, h)
    dmodel = {"stereo": "stereo", "structured_light": "stereo", "tof": "tof"}.get(p.depth_type, "ideal")
    depth = DepthConfig(enabled=True, model=dmodel, baseline_m=p.baseline_m or 0.05,
                        min_z=p.min_z, max_z=p.max_z)
    if dmodel == "tof":
        depth.edge_dropout = 0.3
    return CameraConfig(name=name, preset=f"{key} | {stream}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy,
                        dist=[0.0] * 5, depth=depth)


# --------------------------------------------------------------------------- #
# Calibration import
# --------------------------------------------------------------------------- #
def _as_K(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 9:
        return a.reshape(3, 3)
    if a.size == 4:  # fx fy cx cy
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]])
    raise ValueError("Cannot interpret camera matrix")


def _to_T(R=None, t=None, T=None) -> Optional[np.ndarray]:
    if T is not None:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return T
    if R is None or t is None:
        return None
    R = np.asarray(R, dtype=np.float64)
    if R.size == 3:
        import cv2
        R, _ = cv2.Rodrigues(R.reshape(3, 1))
    out = np.eye(4)
    out[:3, :3] = R.reshape(3, 3)
    out[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return out


def _cam(name, K, dist, w, h, T_world_cam=None, src="") -> CameraConfig:
    dist = [float(x) for x in np.asarray(dist if dist is not None else [], dtype=np.float64).reshape(-1)]
    if len(dist) < 5:
        dist = dist + [0.0] * (5 - len(dist))
    elif 5 < len(dist) < 8:     # OpenCV accepts 4, 5, 8 ... coefficients: 6 or 7 would crash the renderer
        dist = dist + [0.0] * (8 - len(dist))
    c = CameraConfig(name=name, preset="calibrated", width=int(w), height=int(h),
                     fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                     dist=dist[:8], calibration_file=src)
    if T_world_cam is not None:
        c.world_pose = np.asarray(T_world_cam).tolist()
    return c


def load_calibration(path: str) -> List[CameraConfig]:
    """Import one or more calibrated cameras. Supported:

    * OpenCV FileStorage .yml/.yaml/.xml (camera_matrix|K|cameraMatrix, distortion_coefficients|D|distCoeffs,
      image_width/image_height, optional R/T or rvec/tvec = world->camera extrinsics)
    * ROS camera_info YAML
    * Kalibr camchain YAML (multi-camera, builds a rig from T_cn_cnm1)
    * RealSense intrinsics JSON (fx, fy, ppx, ppy, width, height, coeffs) incl. pyrealsense dumps
    * plain JSON: {fx,fy,cx,cy,width,height,dist} or {K, dist, width, height, T_world_cam}
      or a list / {"cameras": [...]} of those.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    src = str(p)
    # -------- OpenCV FileStorage (has %YAML:1.0 header or is xml)
    if p.suffix.lower() == ".xml" or text.lstrip().startswith("%YAML"):
        import cv2
        # parse from memory: OpenCV cannot open paths with non-ASCII characters on Windows
        fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)

        def get(*names):
            for n in names:
                node = fs.getNode(n)
                if not node.empty():
                    return node
            return None

        Kn = get("camera_matrix", "K", "cameraMatrix", "M1", "intrinsic", "Camera_Matrix")
        if Kn is None:
            raise ValueError("No camera matrix in OpenCV file")
        K = Kn.mat()
        Dn = get("distortion_coefficients", "D", "distCoeffs", "D1", "dist_coeffs", "Distortion_Coefficients")
        D = Dn.mat() if Dn is not None else np.zeros(5)
        wn, hn = get("image_width", "width"), get("image_height", "height")
        w = int(wn.real()) if wn is not None else int(round(K[0, 2] * 2 + 1))
        h = int(hn.real()) if hn is not None else int(round(K[1, 2] * 2 + 1))
        Rn, Tn = get("R", "rvec", "rotation"), get("T", "tvec", "translation")
        T_wc = None
        if Rn is not None and Tn is not None:
            T_cw = _to_T(Rn.mat(), Tn.mat())
            T_wc = np.linalg.inv(T_cw)
        fs.release()
        return [_cam(p.stem, K, D, w, h, T_wc, src)]

    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    # -------- Kalibr camchain
    if isinstance(data, dict) and any(k.startswith("cam") and isinstance(v, dict) and "intrinsics" in v
                                      for k, v in data.items()):
        cams, T_c0_ci = [], np.eye(4)
        for k in sorted([k for k in data if k.startswith("cam")], key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0)):
            v = data[k]
            fu, fv, pu, pv = v["intrinsics"][:4]
            K = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1]])
            w, h = v.get("resolution", [int(2 * pu), int(2 * pv)])
            dist = v.get("distortion_coeffs", [])
            if v.get("distortion_model", "radtan") == "radtan":
                d = list(dist) + [0.0] * (4 - len(dist))
                dist = [d[0], d[1], d[2], d[3], 0.0]
            else:  # equidistant etc. -> approximate by zero (warn in name)
                dist = [0.0] * 5
            if "T_cn_cnm1" in v:
                # T_cn_cnm1 maps points of cam n-1 into cam n  -> chain
                T_cn_cnm1 = np.asarray(v["T_cn_cnm1"], dtype=np.float64)
                T_c0_ci = T_c0_ci @ np.linalg.inv(T_cn_cnm1)
            c = _cam(k, K, dist, w, h, None, src)
            _set_rig_pose(c, T_c0_ci)
            cams.append(c)
        return cams

    # -------- lists of cameras
    if isinstance(data, dict) and "cameras" in data:
        data = data["cameras"]
    if isinstance(data, list):
        out = []
        for i, d in enumerate(data):
            out.extend(_from_dict(d, f"cam{i}", src))
        return out
    return _from_dict(data, p.stem, src)


def _set_rig_pose(c: CameraConfig, T_rig_cam: np.ndarray) -> None:
    from scipy.spatial.transform import Rotation
    c.rig_translation = [float(x) for x in T_rig_cam[:3, 3]]
    c.rig_rotation_deg = [float(x) for x in Rotation.from_matrix(T_rig_cam[:3, :3]).as_euler("xyz", degrees=True)]


def _from_dict(d: dict, name: str, src: str) -> List[CameraConfig]:
    # ROS camera_info
    if "camera_matrix" in d and isinstance(d["camera_matrix"], dict):
        K = _as_K(d["camera_matrix"]["data"])
        D = d.get("distortion_coefficients", {}).get("data", [])
        return [_cam(d.get("camera_name", name), K, D, d["image_width"], d["image_height"], None, src)]
    # ROS2 CameraInfo message dumps (k, d, width, height)
    if "k" in d and "width" in d:
        return [_cam(name, _as_K(d["k"]), d.get("d", []), d["width"], d["height"], None, src)]
    # pyrealsense / rs-enumerate style
    if "ppx" in d:
        K = np.array([[d["fx"], 0, d["ppx"]], [0, d["fy"], d["ppy"]], [0, 0, 1]])
        return [_cam(d.get("name", name), K, d.get("coeffs", []), d["width"], d["height"], None, src)]
    if "fx" in d:
        K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]])
    elif "K" in d or "camera_matrix" in d or "intrinsics" in d:
        K = _as_K(d.get("K", d.get("camera_matrix", d.get("intrinsics"))))
    else:
        raise ValueError("Unrecognised calibration JSON/YAML structure")
    w = d.get("width", d.get("image_width"))
    h = d.get("height", d.get("image_height"))
    if w is None or h is None:
        w, h = int(round(2 * K[0, 2] + 1)), int(round(2 * K[1, 2] + 1))
    dist = d.get("dist", d.get("distortion", d.get("coeffs", d.get("D", []))))
    T_wc = None
    if "T_world_cam" in d:
        T_wc = _to_T(T=d["T_world_cam"])
    elif "T_cam_world" in d:
        T_wc = np.linalg.inv(_to_T(T=d["T_cam_world"]))
    elif "extrinsics" in d:
        T_wc = np.linalg.inv(_to_T(T=d["extrinsics"]))  # world->cam convention
    c = _cam(d.get("name", name), K, dist, w, h, T_wc, src)
    if "T_rig_cam" in d:
        _set_rig_pose(c, _to_T(T=d["T_rig_cam"]))
    else:   # written by save_calibration_json
        if d.get("rig_translation") is not None:
            c.rig_translation = [float(x) for x in d["rig_translation"]][:3]
        if d.get("rig_rotation_deg") is not None:
            c.rig_rotation_deg = [float(x) for x in d["rig_rotation_deg"]][:3]
    return [c]


def save_calibration_json(cams: List[CameraConfig], path: str) -> None:
    out = []
    for c in cams:
        out.append(dict(name=c.name, width=c.width, height=c.height, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
                        dist=c.dist, T_world_cam=c.world_pose, rig_translation=c.rig_translation,
                        rig_rotation_deg=c.rig_rotation_deg))
    Path(path).write_text(json.dumps({"cameras": out}, indent=2))
