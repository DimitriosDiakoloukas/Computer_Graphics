import numpy as np
from vector_interp import vector_interp

def t_shading(img, vertices, uv, textImg):
    """
    Texture shading of a triangle using scanline interpolation with vector_interp.
    Each point inside the triangle is shaded using UVs linearly interpolated across scanlines.
    """
    updated_img = np.copy(img)
    H, W, _ = textImg.shape

    # Sort vertices by y-coordinate (ascending)
    inds = np.argsort(vertices[:, 1])
    v = vertices[inds]
    uv = uv[inds]

    # Triangle vertices: v0 (top), v1 (middle), v2 (bottom)
    v0, v1, v2 = v
    uv0, uv1, uv2 = uv

    # Bounding box
    min_y = max(int(np.ceil(v0[1])), 0)
    max_y = min(int(np.floor(v2[1])), img.shape[0] - 1)

    for y in range(min_y, max_y + 1):
        # Determine which edges are intersected at scanline y
        if y < v1[1]:
            # Upper part of triangle
            a = vector_interp(v0, v1, v0, v1, y, dim=2)
            b = vector_interp(v0, v2, v0, v2, y, dim=2)
            uv_a = vector_interp(v0, v1, uv0, uv1, y, dim=2)
            uv_b = vector_interp(v0, v2, uv0, uv2, y, dim=2)
        else:
            # Lower part of triangle
            a = vector_interp(v1, v2, v1, v2, y, dim=2)
            b = vector_interp(v0, v2, v0, v2, y, dim=2)
            uv_a = vector_interp(v1, v2, uv1, uv2, y, dim=2)
            uv_b = vector_interp(v0, v2, uv0, uv2, y, dim=2)

        # Convert to 1D arrays
        a = np.array(a)
        b = np.array(b)
        uv_a = np.array(uv_a)
        uv_b = np.array(uv_b)

        # Ensure a.x <= b.x
        if a[0] > b[0]:
            a, b = b, a
            uv_a, uv_b = uv_b, uv_a

        min_x = max(int(np.ceil(a[0])), 0)
        max_x = min(int(np.floor(b[0])), img.shape[1] - 1)

        for x in range(min_x, max_x + 1):
            uv_p = vector_interp(a, b, uv_a, uv_b, x, dim=1)

            # Map UVs to texture pixel coordinates
            u = int(uv_p[0] * (W - 1))
            v_t = int((1 - uv_p[1]) * (H - 1))  # flip V

            u = np.clip(u, 0, W - 1)
            v_t = np.clip(v_t, 0, H - 1)

            color = textImg[v_t, u]
            updated_img[y, x] = color

    return updated_img
