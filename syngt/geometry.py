"""Small geometry helpers. Conventions:

* world frame: z up, metres, support plane at z = 0
* camera frame: OpenCV (x right, y down, z forward)
* T_a_b maps points from frame b into frame a (4x4)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])


def make_T(R: np.ndarray | None = None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def euler_T(xyz_deg, t=(0, 0, 0)) -> np.ndarray:
    return make_T(Rotation.from_euler("xyz", xyz_deg, degrees=True).as_matrix(), t)


def look_at_cv(eye, target, up=(0, 0, 1), roll_deg: float = 0.0) -> np.ndarray:
    """T_world_cam for an OpenCV camera at ``eye`` looking at ``target``."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z) + 1e-12
    if abs(np.dot(z, up / np.linalg.norm(up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    if roll_deg:
        R = R @ Rotation.from_euler("z", roll_deg, degrees=True).as_matrix()
    return make_T(R, eye)


def gl_projection_from_K(K: np.ndarray, w: int, h: int, near: float, far: float) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4))
    P[0, 0] = 2 * fx / w
    P[0, 2] = 1 - 2 * (cx + 0.5) / w
    P[1, 1] = 2 * fy / h
    P[1, 2] = 2 * (cy + 0.5) / h - 1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    return P


def gl_view_from_cv(T_world_cam: np.ndarray) -> np.ndarray:
    return CV2GL @ inv_T(T_world_cam)


def gl_look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def gl_ortho(l, r, b, t, n, f) -> np.ndarray:
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    M[0, 3], M[1, 3], M[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return M


def gl_perspective(fovy_deg, aspect, n, f) -> np.ndarray:
    t = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1] = t / aspect, t
    M[2, 2], M[2, 3], M[3, 2] = -(f + n) / (f - n), -2 * f * n / (f - n), -1
    return M


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def project(K: np.ndarray, P_cam: np.ndarray, dist=None) -> np.ndarray:
    """Project camera-frame points (N,3) to pixels (N,2). Optional OpenCV distortion."""
    P_cam = np.asarray(P_cam, dtype=np.float64)
    if dist is not None and np.any(np.abs(dist) > 1e-12):
        import cv2
        uv, _ = cv2.projectPoints(P_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(dist, np.float64))
        return uv.reshape(-1, 2)
    z = np.clip(P_cam[:, 2:3], 1e-9, None)
    uv = P_cam[:, :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    return Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()


def rot_z(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    if max_deg <= 0:
        return np.eye(3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * math.radians(rng.uniform(-max_deg, max_deg))).as_matrix()
