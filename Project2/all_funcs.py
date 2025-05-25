import numpy as np
from render_utils import render_img
from typing import Tuple

def translate(t_vec: np.ndarray) -> np.ndarray:
    t_vec = np.asarray(t_vec, dtype=float).reshape(-1)
    if t_vec.size != 3:
        raise ValueError("t_vec must have 3 elements")
    xform = np.eye(4)
    xform[:3, 3] = t_vec
    return xform

def rotate(axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size != 3:
        raise ValueError("axis must have 3 elements")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("axis length must not be zero")
    axis = axis / norm
    center = np.asarray(center, dtype=float).reshape(-1)
    if center.size != 3:
        raise ValueError("center must have 3 elements")
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    t = center - R @ center
    xform = np.eye(4)
    xform[:3, :3] = R
    xform[:3, 3] = t
    return xform

def compose(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    mat1 = np.asarray(mat1, dtype=float)
    mat2 = np.asarray(mat2, dtype=float)
    if mat1.shape != (4, 4) or mat2.shape != (4, 4):
        raise ValueError("inputs must be 4x4 matrices")
    return mat2 @ mat1

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    R = np.asarray(R)
    c0 = np.asarray(c0).reshape(3, 1)
    
    if pts.shape[0] != 3:
        pts = pts.T
        if pts.shape[0] != 3:
            raise ValueError("pts must be 3xN or Nx3 array")
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3 rotation matrix")
    if c0.shape != (3, 1):
        raise ValueError("c0 must be 3D vector")
    transformed = R @ (pts - c0)
    
    return transformed.T

def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int, res_w: int, res_h: int) -> np.ndarray:
    pts_2d = np.asarray(pts_2d)
    if pts_2d.shape[0] != 2:
        raise ValueError("Expected pts_2d to be shape (2, N)")

    scale_x = res_w / plane_w
    scale_y = res_h / plane_h

    center_x = res_w / 2
    center_y = res_h / 2

    N = pts_2d.shape[1]
    pixel_coords = np.zeros((2, N), dtype=int)

    for i in range(N):
        x_cam = pts_2d[0, i]
        y_cam = pts_2d[1, i]

        pixel_x = x_cam * scale_x + center_x
        pixel_y = -y_cam * scale_y + center_y

        pixel_x = np.clip(np.round(pixel_x), 0, res_w - 1)
        pixel_y = np.clip(np.round(pixel_y), 0, res_h - 1)

        pixel_coords[0, i] = pixel_x
        pixel_coords[1, i] = pixel_y

    return pixel_coords

def lookat(eye: np.ndarray, up: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(eye).reshape(3,)
    up = np.asarray(up).reshape(3,)
    target = np.asarray(target).reshape(3,)

    z_axis = (target - eye)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(x_axis, z_axis)

    R = np.stack((x_axis, y_axis, -z_axis), axis=1)

    return R, eye

def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray, t: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
    transforms = world2view(pts, R, t)
    xQ = []
    yQ = []
    depths = []

    for point in transforms:
        depths.append(point[2])

    depths = np.array(depths)
    
    for i in range(len(depths)):
        xQ.append((focal / depths[i]) * transforms[i][0])
        yQ.append((focal / depths[i]) * transforms[i][1])
    
    xQ = np.array(xQ)
    yQ = np.array(yQ)
    pts_2d = np.array([xQ, yQ])
    
    return pts_2d, depths


def render_object(v_pos, v_clr, v_uvs, t_pos_idx, plane_h, plane_w, res_h, res_w, focal, eye, up, target, texImg) -> np.ndarray:
    image = np.ones((res_h, res_w, 3), dtype=np.float32)

    R, t = lookat(eye, up, target)
    projected_pts, depth = perspective_project(v_pos.T, focal, R, t)
    pixel_coords = rasterize(projected_pts, plane_w, plane_h, res_w, res_h).T
    vertices_2d = np.hstack([pixel_coords, depth[:, None]])

    image = render_img(t_pos_idx, vertices_2d, v_clr, v_uvs, depth.T.flatten(), "t", texImg)

    return image
