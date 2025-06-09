import numpy as np
from typing import Tuple

def world2view(pts: np.ndarray, R: np.ndarray, c0: np.ndarray) -> np.ndarray:
    """
    Transform points from world coordinates to view coordinates using a rotation matrix and a translation vector.
    Args:
        pts (np.ndarray): A 3xN or Nx3 array of points in world coordinates.
        R (np.ndarray): A 3x3 rotation matrix.
        c0 (np.ndarray): A 3D vector representing the camera position in world coordinates.
    Returns:
        np.ndarray: A 3xN array of points in view coordinates.           
    """
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
    """
    Convert 2D points in camera coordinates to pixel coordinates in the image plane.
    Args:
        pts_2d (np.ndarray): A 2xN array of points in camera coordinates.
        plane_w (int): The width of the image plane.
        plane_h (int): The height of the image plane.
        res_w (int): The width of the resulting image.
        res_h (int): The height of the resulting image.
    Returns:
        np.ndarray: A 2xN array of pixel coordinates in the resulting image.
    """
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
    """
    Create a rotation matrix and translation vector for a camera looking at a target point.
    Args:
        eye (np.ndarray): A 3D vector representing the camera position.
        up (np.ndarray): A 3D vector representing the up direction of the camera.
        target (np.ndarray): A 3D vector representing the target point the camera is looking at.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the 3x3 rotation matrix and the camera position as a 3D vector.
    """
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
    """
    Project 3D points onto a 2D plane using perspective projection.
        Args:
        pts (np.ndarray): A 3xN or Nx3 array of points in world coordinates.
        focal (float): The focal length of the camera.
        R (np.ndarray): A 3x3 rotation matrix.
        t (np.ndarray): A 3D vector representing the camera position in world coordinates.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing a 2xN array of projected 2D points and a 1D array of depths.
    """
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