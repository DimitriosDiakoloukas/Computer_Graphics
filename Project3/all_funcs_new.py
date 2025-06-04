import numpy as np
from typing import Union, List
from all_funcs_prev import perspective_project, lookat, rasterize
from vector_interp import vector_interp
from MatPhong import light

def calc_normals(pts: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    """
    Computes the unit normal vector for each vertex of the mesh,
    oriented outward using the right-hand rule.

    Parameters
    ----------
    pts : (3, N_v) ndarray
        Coordinates of all vertices.
    t_pos_idx : (3, N_T) ndarray (int)
        Vertex indices for each triangle (as columns).
        Can be 0-based (Python style) or 1-based (as in theory).

    Returns
    -------
    nrm : (3, N_v) ndarray
        Unit normal vectors for each vertex.
    """

    # --- Initialize normal vector array (we will accumulate per vertex)
    nrm = np.zeros_like(pts)           # 3 × N_v

    # --- Convert indices to 0-based if they are 1-based
    if t_pos_idx.min() == 1:
        tri_idx = t_pos_idx.astype(int) - 1
    else:
        tri_idx = t_pos_idx.astype(int)

    # --- Loop over each triangle: accumulate face normal to each vertex
    for k in range(tri_idx.shape[1]):
        i0, i1, i2 = tri_idx[:, k]     # indices of the triangle's three vertices

        # Coordinates of the triangle's vertices
        v0, v1, v2 = pts[:, i0], pts[:, i1], pts[:, i2]

        # Edges (v1-v0, v2-v0) and triangle normal (right-hand rule)
        e1 = v1 - v0
        e2 = v2 - v0
        face_n = np.cross(e1, e2)      # unnormalized face normal

        # Add the face normal to the three vertices
        nrm[:, i0] += face_n
        nrm[:, i1] += face_n
        nrm[:, i2] += face_n

    # --- Normalize per-vertex normals
    lengths = np.linalg.norm(nrm, axis=0)
    non_zero = lengths > 0
    nrm[:, non_zero] /= lengths[non_zero]

    return nrm


def shade_gouraud(
    v_proj: np.ndarray,      # (2, 3) screen-space coords
    v_world: np.ndarray,     # (3, 3) world-space coords for lighting
    v_nrm: np.ndarray,
    v_uvs: np.ndarray,
    tex: np.ndarray,
    cam_pos: np.ndarray,
    mat,
    l_pos, l_int, l_amb,
    img: np.ndarray
) -> np.ndarray:
    updated_img = np.copy(img)

    vertex_colors = []
    for i in range(3):
        pt = v_world[:, i]
        nrm = v_nrm[:, i]
        uv = v_uvs[:, i]
        tex_color = tex[int((1 - uv[1]) * (tex.shape[0] - 1)), int(uv[0] * (tex.shape[1] - 1))]
        col = light(pt, nrm, tex_color, cam_pos, mat, l_pos, l_int, l_amb)
        vertex_colors.append(col)
    vertex_colors = np.array(vertex_colors)

    screen_coords = np.vstack([v_proj, np.zeros(3)])  # add dummy depth
    indices = np.argsort(screen_coords[1])
    v = screen_coords[:, indices].T
    c = vertex_colors[indices]
    uv = v_uvs[:, indices].T

    min_y = max(int(np.ceil(v[0][1])), 0)
    max_y = min(int(np.floor(v[2][1])), img.shape[0] - 1)

    for y in range(min_y, max_y + 1):
        if y < v[1][1]:
            a = vector_interp(v[0], v[1], v[0], v[1], y, dim=2)
            b = vector_interp(v[0], v[2], v[0], v[2], y, dim=2)
            ca = vector_interp(v[0], v[1], c[0], c[1], y, dim=2)
            cb = vector_interp(v[0], v[2], c[0], c[2], y, dim=2)
        else:
            a = vector_interp(v[1], v[2], v[1], v[2], y, dim=2)
            b = vector_interp(v[0], v[2], v[0], v[2], y, dim=2)
            ca = vector_interp(v[1], v[2], c[1], c[2], y, dim=2)
            cb = vector_interp(v[0], v[2], c[0], c[2], y, dim=2)

        a = np.array(a)
        b = np.array(b)
        ca = np.array(ca)
        cb = np.array(cb)

        if a[0] > b[0]:
            a, b = b, a
            ca, cb = cb, ca

        min_x = max(int(np.ceil(a[0])), 0)
        max_x = min(int(np.floor(b[0])), img.shape[1] - 1)

        for x in range(min_x, max_x + 1):
            c_p = vector_interp(a, b, ca, cb, x, dim=1)
            updated_img[y, x] = np.clip(c_p, 0, 1)

    return updated_img


def shade_phong(
    v_proj: np.ndarray,      # (2, 3) screen coords
    v_world: np.ndarray,     # (3, 3) 3D positions
    v_nrm: np.ndarray,       # (3, 3)
    v_uvs: np.ndarray,       # (2, 3)
    tex: np.ndarray,
    cam_pos: np.ndarray,
    mat,
    l_pos, l_int, l_amb,
    img: np.ndarray
) -> np.ndarray:
    updated_img = np.copy(img)
    H, W, _ = tex.shape

    screen_coords = np.vstack([v_proj, np.zeros(3)])
    norms = v_nrm.T
    uvs = v_uvs.T
    world = v_world.T

    inds = np.argsort(screen_coords[1])
    v = screen_coords[:, inds].T
    n = norms[inds]
    uv = uvs[inds]
    wp = world[inds]

    min_y = max(int(np.ceil(v[0][1])), 0)
    max_y = min(int(np.floor(v[2][1])), img.shape[0] - 1)

    for y in range(min_y, max_y + 1):
        if y < v[1][1]:
            a = vector_interp(v[0], v[1], v[0], v[1], y, dim=2)
            b = vector_interp(v[0], v[2], v[0], v[2], y, dim=2)
            na = vector_interp(v[0], v[1], n[0], n[1], y, dim=2)
            nb = vector_interp(v[0], v[2], n[0], n[2], y, dim=2)
            uva = vector_interp(v[0], v[1], uv[0], uv[1], y, dim=2)
            uvb = vector_interp(v[0], v[2], uv[0], uv[2], y, dim=2)
            wpa = vector_interp(v[0], v[1], wp[0], wp[1], y, dim=2)
            wpb = vector_interp(v[0], v[2], wp[0], wp[2], y, dim=2)
        else:
            a = vector_interp(v[1], v[2], v[1], v[2], y, dim=2)
            b = vector_interp(v[0], v[2], v[0], v[2], y, dim=2)
            na = vector_interp(v[1], v[2], n[1], n[2], y, dim=2)
            nb = vector_interp(v[0], v[2], n[0], n[2], y, dim=2)
            uva = vector_interp(v[1], v[2], uv[1], uv[2], y, dim=2)
            uvb = vector_interp(v[0], v[2], uv[0], uv[2], y, dim=2)
            wpa = vector_interp(v[1], v[2], wp[1], wp[2], y, dim=2)
            wpb = vector_interp(v[0], v[2], wp[0], wp[2], y, dim=2)

        a = np.array(a)
        b = np.array(b)
        na = np.array(na)
        nb = np.array(nb)
        uva = np.array(uva)
        uvb = np.array(uvb)
        wpa = np.array(wpa)
        wpb = np.array(wpb)

        if a[0] > b[0]:
            a, b = b, a
            na, nb = nb, na
            uva, uvb = uvb, uva
            wpa, wpb = wpb, wpa

        min_x = max(int(np.ceil(a[0])), 0)
        max_x = min(int(np.floor(b[0])), img.shape[1] - 1)

        for x in range(min_x, max_x + 1):
            p_nrm = vector_interp(a, b, na, nb, x, dim=1)
            p_uv = vector_interp(a, b, uva, uvb, x, dim=1)
            p_wpos = vector_interp(a, b, wpa, wpb, x, dim=1)

            p_nrm = np.array(p_nrm)
            p_uv = np.array(p_uv)
            p_wpos = np.array(p_wpos)

            u = int(p_uv[0] * (W - 1))
            v_t = int((1 - p_uv[1]) * (H - 1))
            u = np.clip(u, 0, W - 1)
            v_t = np.clip(v_t, 0, H - 1)
            tex_color = tex[v_t, u]

            color = light(p_wpos, p_nrm, tex_color, cam_pos, mat, l_pos, l_int, l_amb)
            updated_img[y, x] = color

    return updated_img


def render_object(
    v_pos: np.ndarray,
    v_uvs: np.ndarray,
    t_pos_idx: np.ndarray,
    tex: np.ndarray,
    plane_h: int,
    plane_w: int,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    mat,
    l_pos: Union[np.ndarray, List[np.ndarray]],
    l_int: Union[np.ndarray, List[np.ndarray]],
    l_amb: np.ndarray,
    shader: str
) -> np.ndarray:
    print("\n========== render_object DEBUG ==========")

    print("[Step 1] Computing normals...")
    v_norm = calc_normals(v_pos, t_pos_idx)
    print(f"Normals shape: {v_norm.shape}")

    print("[Step 2] Projecting vertices...")
    R, t = lookat(eye, up, target)
    projected_2d, depths = perspective_project(v_pos, focal, R, t)
    pixel_coords = rasterize(projected_2d, plane_w, plane_h, res_w, res_h)
    print(f"Projected 2D shape: {projected_2d.shape}")
    print(f"Rasterized coords min: {pixel_coords.min(axis=1)} max: {pixel_coords.max(axis=1)}")

    img = np.zeros((res_h, res_w, 3), dtype=np.float32)

    print(f"[Step 3] Shading {t_pos_idx.shape[1]} triangles using '{shader}' shader...")
    for k in range(t_pos_idx.shape[1]):
        indices = t_pos_idx[:, k]

        tri_proj = pixel_coords[:, indices]    # (2, 3)
        tri_world = v_pos[:, indices]          # (3, 3)
        tri_norm = v_norm[:, indices]          # (3, 3)
        tri_uvs = v_uvs[:, indices]            # (2, 3)

        if shader == "phong":
            img = shade_phong(
                tri_proj, tri_world, tri_norm, tri_uvs, tex, eye,
                mat, l_pos, l_int, l_amb, img
            )
        elif shader == "gouraud":
            img = shade_gouraud(
                tri_proj, tri_world, tri_norm, tri_uvs, tex, eye,
                mat, l_pos, l_int, l_amb, img
            )
        else:
            raise ValueError("shader must be 'phong' or 'gouraud'")

    return np.clip(img, 0, 1)

