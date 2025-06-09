import numpy as np
from typing import Union, List

class MatPhong:
    def __init__(self, ka: float, kd: float, ks: float, n: float) -> None:
        self.ka = ka  # Ambient coefficient
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.n = n    # Phong exponent


def light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb):
    """
    Calculate the Phong lighting model at a point in space.
    Args:
        pt (np.ndarray): The point in space where the lighting is calculated (3D vector).
        nrm (np.ndarray): The normal vector at the point (3D vector).
        vclr (np.ndarray): The vertex color at the point (3D vector).
        cam_pos (np.ndarray): The camera position (3D vector).
        mat (MatPhong): Material properties including ambient, diffuse, and specular coefficients.
        l_pos (Union[np.ndarray, List[np.ndarray]]): Light positions (can be a single light or a list of lights).
        l_int (Union[float, List[float]]): Light intensities corresponding to the light positions.
        l_amb (float): Ambient light intensity.
    Returns:
        np.ndarray: The resulting color at the point, calculated using the Phong lighting model.     
    """
    result = np.zeros(3)
    nrm = nrm / np.linalg.norm(nrm)  # normalize normal

    # Ambient term
    result += mat.ka * l_amb * vclr

    # Support both single and multiple lights
    if isinstance(l_pos, list):
        light_positions = l_pos
        light_intensities = l_int
    else:
        light_positions = [l_pos]
        light_intensities = [l_int]

    # Iterate over each light source
    for i in range(len(light_positions)):
        L = light_positions[i] - pt
        L_norm = np.linalg.norm(L)
        if L_norm == 0:
            continue
        L = L / L_norm

        V = cam_pos - pt
        V_norm = np.linalg.norm(V)
        if V_norm == 0:
            continue
        V = V / V_norm

        R = 2 * np.dot(nrm, L) * nrm - L
        R = R / np.linalg.norm(R)

        # Calculate diffuse and specular components
        # Ensure that the dot products are non-negative
        diff = mat.kd * max(np.dot(nrm, L), 0.0)
        spec = mat.ks * max(np.dot(R, V), 0.0) ** mat.n

        result += light_intensities[i] * vclr * diff
        result += light_intensities[i] * spec

    return np.clip(result, 0, 1)
