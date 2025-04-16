import numpy as np
from vector_interp import vector_interp


# def t_shading(img, vertices, uv, textImg):
#     """
#     Texture shading of a triangle using barycentric interpolation of UVs.

#     Parameters:
#         img (np.ndarray): Canvas image (M x N x 3), will be updated.
#         vertices (np.ndarray): 3x2 array of the triangle's [x, y] screen coordinates.
#         uv (np.ndarray): 3x2 array of UV coordinates (u, v) in [0, 1] corresponding to each vertex.
#         textImg (np.ndarray): The texture image (H x W x 3), RGB float32 in [0, 1].

#     Returns:
#         updated_img (np.ndarray): Image with triangle shaded using texture mapping.
#     """
#     updated_img = np.copy(img)

#     H, W, _ = textImg.shape

#     # Bounding box for the triangle
#     x = vertices[:, 0]
#     y = vertices[:, 1]
#     min_x = max(int(np.floor(np.min(x))), 0)
#     max_x = min(int(np.ceil(np.max(x))), img.shape[1] - 1)
#     min_y = max(int(np.floor(np.min(y))), 0)
#     max_y = min(int(np.ceil(np.max(y))), img.shape[0] - 1)

#     # Setup vertices
#     v0, v1, v2 = vertices
#     uv0, uv1, uv2 = uv

#     # Precompute area for barycentric
#     def edge_fn(a, b, c):
#         return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

#     area = edge_fn(v0, v1, v2)
#     if area == 0:
#         return updated_img  # skip degenerate triangle

#     # Loop over pixels in bounding box
#     for j in range(min_y, max_y + 1):
#         for i in range(min_x, max_x + 1):
#             p = np.array([i + 0.5, j + 0.5])

#             w0 = edge_fn(v1, v2, p)
#             w1 = edge_fn(v2, v0, p)
#             w2 = edge_fn(v0, v1, p)

#             if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
#                 # Barycentric coordinates (normalized)
#                 alpha = w0 / area
#                 beta = w1 / area
#                 gamma = w2 / area

#                 # Interpolate UV
#                 uv_p = alpha * uv0 + beta * uv1 + gamma * uv2

#                 # Convert UV (0-1) to texture coordinates
#                 u = int(uv_p[0] * (W - 1))
#                 v = int((1 - uv_p[1]) * (H - 1))  # flip v-axis

#                 # Clamp to valid texture range
#                 u = np.clip(u, 0, W - 1)
#                 v = np.clip(v, 0, H - 1)

#                 # Sample color and update pixel
#                 color = textImg[v, u]
#                 updated_img[j, i] = color

#     return updated_img


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
