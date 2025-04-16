import numpy as np
from vector_interp import vector_interp

# def f_shading(img, vertices, vcolors):
#     """
#     Flat shading of a triangle: fill the triangle with the average color of its vertices.
#     """
#     updated_img = np.copy(img)

#     x = vertices[:, 0]
#     y = vertices[:, 1]

#     min_x = max(int(np.floor(np.min(x))), 0)
#     max_x = min(int(np.ceil(np.max(x))), img.shape[1] - 1)
#     min_y = max(int(np.floor(np.min(y))), 0)
#     max_y = min(int(np.ceil(np.max(y))), img.shape[0] - 1)

#     flat_color = np.mean(vcolors, axis=0)

#     def edge_fn(v0, v1, p):
#         return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

#     v0, v1, v2 = vertices

#     for j in range(min_y, max_y + 1):
#         for i in range(min_x, max_x + 1):
#             p = np.array([i + 0.5, j + 0.5])
#             w0 = edge_fn(v1, v2, p)
#             w1 = edge_fn(v2, v0, p)
#             w2 = edge_fn(v0, v1, p)

#             if w0 >= 0 and w1 >= 0 and w2 >= 0:
#                 updated_img[j, i] = flat_color

#     return updated_img

def f_shading(img, vertices, vcolors):
    """
    Flat shading using scanline polygon filling.
    Fills the triangle with the average of its vertex colors.
    """
    updated_img = np.copy(img)
    flat_color = np.mean(vcolors, axis=0)

    # Sort vertices by y-coordinate
    inds = np.argsort(vertices[:, 1])
    v = vertices[inds]

    v0, v1, v2 = v

    min_y = max(int(np.ceil(v0[1])), 0)
    max_y = min(int(np.floor(v2[1])), img.shape[0] - 1)

    for y in range(min_y, max_y + 1):
        if y < v1[1]:
            a = vector_interp(v0, v1, v0, v1, y, dim=2)
            b = vector_interp(v0, v2, v0, v2, y, dim=2)
        else:
            a = vector_interp(v1, v2, v1, v2, y, dim=2)
            b = vector_interp(v0, v2, v0, v2, y, dim=2)

        a = np.array(a)
        b = np.array(b)

        if a[0] > b[0]:
            a, b = b, a

        min_x = max(int(np.ceil(a[0])), 0)
        max_x = min(int(np.floor(b[0])), img.shape[1] - 1)

        for x in range(min_x, max_x + 1):
            updated_img[y, x] = flat_color

    return updated_img
