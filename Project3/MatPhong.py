import numpy as np
from typing import Union, List

class MatPhong:
    def __init__(self, ka: float, kd: float, ks: float, n: float) -> None:
        self.ka = ka  # Ambient coefficient
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.n = n    # Phong exponent


def light(pt, nrm, vclr, cam_pos, mat, l_pos, l_int, l_amb):
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

        diff = mat.kd * max(np.dot(nrm, L), 0.0)
        spec = mat.ks * max(np.dot(R, V), 0.0) ** mat.n

        result += light_intensities[i] * vclr * diff
        result += light_intensities[i] * spec

    return np.clip(result, 0, 1)
