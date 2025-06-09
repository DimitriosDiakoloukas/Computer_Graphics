import numpy as np
from typing import Union, List
from all_funcs_prev import perspective_project, lookat, rasterize
from vector_interp import vector_interp
from MatPhong import light

DBG = True

def calc_normals(pts: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    """
    Calculate normals for a set of points based on triangle indices.
    Args:
        pts (np.ndarray): Points in shape (3, N) where N is the number of points.
        t_pos_idx (np.ndarray): Triangle indices in shape (3, M) where M is the number of triangles.
    Returns:
        np.ndarray: Normals in shape (3, N) where N is the number of points.
    Raises:
        ValueError: If pts is not of shape (3, N) or t_pos_idx is not of shape (3, M).
    """
    nrm = np.zeros_like(pts)
    if t_pos_idx.min() == 1:
        tri_idx = t_pos_idx.astype(int) - 1
    else:
        tri_idx = t_pos_idx.astype(int)

    for k in range(tri_idx.shape[1]):
        i0, i1, i2 = tri_idx[:, k]
        v0 = pts[:, i0]
        v1 = pts[:, i1]
        v2 = pts[:, i2]
        face_n = np.cross(v1 - v0, v2 - v0)
        nrm[:, i0] += face_n
        nrm[:, i1] += face_n
        nrm[:, i2] += face_n

    lengths = np.linalg.norm(nrm, axis=0)
    nonzero = (lengths > 1e-8)
    nrm[:, nonzero] /= lengths[nonzero]

    if DBG:
        print(f"[calc_normals]  produced normals of shape {nrm.shape}")

    return nrm

def shade_gouraud(v_pos, v_nrm, v_uvs, tex, cam_pos, mat, l_pos, l_int, l_amb, img):
    """
    Gouraud shading for a triangle defined by its vertices.
    Args:
        v_pos (np.ndarray): Vertex positions in shape (3, 3).
        v_nrm (np.ndarray): Vertex normals in shape (3, 3).
        v_uvs (np.ndarray): Vertex UV coordinates in shape (2, 3).
        tex (np.ndarray): Texture image in shape (H, W, 3).
        cam_pos (np.ndarray): Camera position in shape (3,).
        mat (MatPhong): Material properties.    
        l_pos (List[np.ndarray]): Light positions in shape (3, N).
        l_int (List[np.ndarray]): Light intensities in shape (3, N).
        l_amb (np.ndarray): Ambient light intensity in shape (3,).
        img (np.ndarray): Image to draw on in shape (H, W, 3).
    Returns:
        np.ndarray: Image with the triangle shaded.
    Raises:
        ValueError: If v_pos, v_nrm, or v_uvs are not of shape (3, 3) or (2, 3).
    """
    if DBG:
        print("  [Gouraud] rasterising one triangle")

    H, W, _ = tex.shape
    out = img
    vcol = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        P_cam = np.array([v_pos[0, i], v_pos[1, i], v_pos[2, i]])
        N_i = v_nrm[:, i]
        uv_i = v_uvs[:, i]
        texel = tex[int((1 - uv_i[1]) * (H - 1)), int(uv_i[0] * (W - 1))]
        vcol[i] = light(P_cam, N_i, texel, cam_pos, mat, l_pos, l_int, l_amb)
        if DBG:
            print(f"    vertex {i}: uv={uv_i}, texel={texel[:3]}, lit‐color={vcol[i]}")

    scr = v_pos[0:2, :]
    order = np.argsort(scr[1, :])
    scr_sorted = scr[:, order].T
    col_sorted = vcol[order]
    uv_sorted = v_uvs[:, order].T

    y0 = int(np.ceil(scr_sorted[0, 1]))
    y2 = int(np.floor(scr_sorted[2, 1]))
    y0 = max(0, y0)
    y2 = min(out.shape[0] - 1, y2)

    # Loop through each scanline
    for yy in range(y0, y2 + 1):
        if yy < scr_sorted[1, 1]:
            # Interpolate between the first two vertices
            A = vector_interp(scr_sorted[0], scr_sorted[1], scr_sorted[0], scr_sorted[1], yy, dim=2)
            B = vector_interp(scr_sorted[0], scr_sorted[2], scr_sorted[0], scr_sorted[2], yy, dim=2)
            CA = vector_interp(scr_sorted[0], scr_sorted[1], col_sorted[0], col_sorted[1], yy, dim=2)
            CB = vector_interp(scr_sorted[0], scr_sorted[2], col_sorted[0], col_sorted[2], yy, dim=2)
        else:
            # If the scanline is below the middle vertex
            A = vector_interp(scr_sorted[1], scr_sorted[2], scr_sorted[1], scr_sorted[2], yy, dim=2)
            B = vector_interp(scr_sorted[0], scr_sorted[2], scr_sorted[0], scr_sorted[2], yy, dim=2)
            CA = vector_interp(scr_sorted[1], scr_sorted[2], col_sorted[1], col_sorted[2], yy, dim=2)
            CB = vector_interp(scr_sorted[0], scr_sorted[2], col_sorted[0], col_sorted[2], yy, dim=2)

        A, B, CA, CB = np.array(A), np.array(B), np.array(CA), np.array(CB)
        if A[0] > B[0]:
            A, B = B, A
            CA, CB = CB, CA

        x0 = int(np.ceil(A[0]))
        x2 = int(np.floor(B[0]))
        x0 = max(0, x0)
        x2 = min(out.shape[1] - 1, x2)

        for xx in range(x0, x2 + 1):
            C_pix = vector_interp(A, B, CA, CB, xx, dim=1)
            out[yy, xx] = np.clip(C_pix, 0, 1)

    return out

def shade_phong(v_pos, v_nrm, v_uvs, tex, cam_pos, mat, l_pos, l_int, l_amb, img):
    """
    Phong shading for a triangle defined by its vertices.
    Args:
        v_pos (np.ndarray): Vertex positions in shape (3, 3).
        v_nrm (np.ndarray): Vertex normals in shape (3, 3).
        v_uvs (np.ndarray): Vertex UV coordinates in shape (2, 3).
        tex (np.ndarray): Texture image in shape (H, W, 3).
        cam_pos (np.ndarray): Camera position in shape (3,).
        mat (MatPhong): Material properties.
        l_pos (List[np.ndarray]): Light positions in shape (3, N).
        l_int (List[np.ndarray]): Light intensities in shape (3, N).
        l_amb (np.ndarray): Ambient light intensity in shape (3,).
        img (np.ndarray): Image to draw on in shape (H, W, 3).
    Returns:    
        np.ndarray: Image with the triangle shaded.
    Raises:
        ValueError: If v_pos, v_nrm, or v_uvs are not of shape (3, 3) or (2, 3).
    """
    if DBG:
        print("  [Phong]     rasterising one triangle")

    out = img
    H, W, _ = tex.shape
    scr = v_pos[0:2, :]
    order = np.argsort(scr[1, :])
    scr_sorted = scr[:, order].T
    n_sorted = v_nrm[:, order].T
    uv_sorted = v_uvs[:, order].T
    Pcam_sorted = np.vstack([v_pos[0, order], v_pos[1, order], v_pos[2, order]]).T

    y0 = int(np.ceil(scr_sorted[0, 1]))
    y2 = int(np.floor(scr_sorted[2, 1]))
    y0 = max(0, y0)
    y2 = min(out.shape[0] - 1, y2)

    for yy in range(y0, y2 + 1):
        if yy < scr_sorted[1, 1]:
            # Interpolate between the first two vertices
            A = vector_interp(scr_sorted[0], scr_sorted[1], scr_sorted[0], scr_sorted[1], yy, dim=2)
            B = vector_interp(scr_sorted[0], scr_sorted[2], scr_sorted[0], scr_sorted[2], yy, dim=2)
            NA = vector_interp(scr_sorted[0], scr_sorted[1], n_sorted[0], n_sorted[1], yy, dim=2)
            NB = vector_interp(scr_sorted[0], scr_sorted[2], n_sorted[0], n_sorted[2], yy, dim=2)
            UVA = vector_interp(scr_sorted[0], scr_sorted[1], uv_sorted[0], uv_sorted[1], yy, dim=2)
            UVB = vector_interp(scr_sorted[0], scr_sorted[2], uv_sorted[0], uv_sorted[2], yy, dim=2)
            PA = vector_interp(scr_sorted[0], scr_sorted[1], Pcam_sorted[0], Pcam_sorted[1], yy, dim=2)
            PB = vector_interp(scr_sorted[0], scr_sorted[2], Pcam_sorted[0], Pcam_sorted[2], yy, dim=2)
        else:
            # If the scanline is below the middle vertex
            A = vector_interp(scr_sorted[1], scr_sorted[2], scr_sorted[1], scr_sorted[2], yy, dim=2)
            B = vector_interp(scr_sorted[0], scr_sorted[2], scr_sorted[0], scr_sorted[2], yy, dim=2)
            NA = vector_interp(scr_sorted[1], scr_sorted[2], n_sorted[1], n_sorted[2], yy, dim=2)
            NB = vector_interp(scr_sorted[0], scr_sorted[2], n_sorted[0], n_sorted[2], yy, dim=2)
            UVA = vector_interp(scr_sorted[1], scr_sorted[2], uv_sorted[1], uv_sorted[2], yy, dim=2)
            UVB = vector_interp(scr_sorted[0], scr_sorted[2], uv_sorted[0], uv_sorted[2], yy, dim=2)
            PA = vector_interp(scr_sorted[1], scr_sorted[2], Pcam_sorted[1], Pcam_sorted[2], yy, dim=2)
            PB = vector_interp(scr_sorted[0], scr_sorted[2], Pcam_sorted[0], Pcam_sorted[2], yy, dim=2)

        A, B = np.array(A), np.array(B)
        NA_, NB_ = np.array(NA), np.array(NB)
        UVA_, UVB_ = np.array(UVA), np.array(UVB)
        PA_, PB_ = np.array(PA), np.array(PB)

        # Ensure A is always the left vertex
        # This ensures that we always interpolate from left to right
        # If A is to the right of B, swap them
        # and also swap the normals, UVs, and positions
        if A[0] > B[0]:
            A, B = B, A
            NA_, NB_ = NB_, NA_
            UVA_, UVB_ = UVB_, UVA_
            PA_, PB_ = PB_, PA_

        x0 = int(np.ceil(A[0]))
        x2 = int(np.floor(B[0]))
        x0 = max(0, x0)
        x2 = min(out.shape[1] - 1, x2)

        for xx in range(x0, x2 + 1):
            # Interpolate the position, normal, UVs, and color
            # for the current pixel (xx, yy)    
            N_px = vector_interp(A, B, NA_, NB_, xx, dim=1)
            UV_px = vector_interp(A, B, UVA_, UVB_, xx, dim=1)
            P_px = vector_interp(A, B, PA_, PB_, xx, dim=1)
            u_pix = int(UV_px[0] * (W - 1))
            v_pix = int((1 - UV_px[1]) * (H - 1))
            u_pix = np.clip(u_pix, 0, W - 1)
            v_pix = np.clip(v_pix, 0, H - 1)
            texel = tex[v_pix, u_pix]
            out[yy, xx] = light(P_px, N_px, texel, cam_pos, mat, l_pos, l_int, l_amb)

    return out


def render_object(v_pos, v_uvs, t_pos_idx, tex, plane_h, plane_w, res_h, res_w,
                  focal, eye, up, target, mat, l_pos, l_int, l_amb, shader):
    """
    Render a 3D object defined by vertex positions, UVs, and triangle indices.  
    Args:
        v_pos (np.ndarray): Vertex positions in shape (3, N).
        v_uvs (np.ndarray): Vertex UV coordinates in shape (2, N).
        t_pos_idx (np.ndarray): Triangle indices in shape (3, M).
        tex (np.ndarray): Texture image in shape (H, W, 3).
        plane_h (float): Height of the rendering plane.
        plane_w (float): Width of the rendering plane.
        res_h (int): Height of the output image.
        res_w (int): Width of the output image.
        focal (float): Focal length for perspective projection.
        eye (np.ndarray): Camera position in shape (3,).
        up (np.ndarray): Up vector for the camera in shape (3,).
        target (np.ndarray): Target point for the camera in shape (3,).
        mat: Material properties for shading.
        l_pos: List of light positions in shape (3, N).
        l_int: List of light intensities in shape (3, N).
        l_amb: Ambient light intensity in shape (3,).
        shader: Shading method to use ('gouraud' or 'phong').
    Returns:
        np.ndarray: Rendered image in shape (res_h, res_w, 3).
    Raises:
        ValueError: If shader is not 'gouraud' or 'phong'.    
    """
    if DBG:
        print("\n========== render_object() ==========")

    # Here v_pos is in shape (3, N) and t_pos_idx is in shape (3, M)
    Wnorm = calc_normals(v_pos, t_pos_idx)
    R, t_vec = lookat(eye, up, target)
    v_nrm = R @ Wnorm

    if DBG:
        print("[Step 1] normals computed, shape =", v_nrm.shape)

    v_cam = (R @ v_pos) + t_vec[:, None]
    proj2d, depth = perspective_project(v_cam, focal, np.eye(3), np.zeros(3))
    pix = rasterize(proj2d, plane_w, plane_h, res_w, res_h)

    if DBG:
        print("[Step 2] projection & rasterise:")
        print("       proj2d shape =", proj2d.shape, "  depth shape =", depth.shape)
        print("       pix range x:[%d..%d], y:[%d..%d]" %
              (pix[0].min(), pix[0].max(), pix[1].min(), pix[1].max()))

    # Ensure pix is in the range [0, res_w-1] and [0, res_h-1]
    img = np.ones((res_h, res_w, 3), dtype=np.float32)
    tri_depth = np.zeros(t_pos_idx.shape[1], dtype=np.float32)
    for k in range(t_pos_idx.shape[1]):
        # Ensure indices are integers and within bounds
        idx = t_pos_idx[:, k].astype(int)
        tri_depth[k] = np.mean(depth[idx])
    order = np.argsort(tri_depth)[::-1]

    if DBG:
        print("[Step 3] sorting triangles by depth, #tris =", len(order))

    for k in order:
        idx = t_pos_idx[:, k].astype(int)
        tri_pix2 = pix[:, idx]
        tri_depths = depth[idx]
        tri_proj3 = np.vstack([tri_pix2, tri_depths])
        tri_uvs = v_uvs[:, idx]
        tri_nrm_cam = v_nrm[:, idx]

        if shader.lower() == "gouraud":
            img = shade_gouraud(tri_proj3, tri_nrm_cam, tri_uvs, tex,
                                eye, mat, l_pos, l_int, l_amb, img)
        elif shader.lower() == "phong":
            img = shade_phong(tri_proj3, tri_nrm_cam, tri_uvs, tex,
                              eye, mat, l_pos, l_int, l_amb, img)
        else:
            raise ValueError("render_object: shader must be 'gouraud' or 'phong'")

    if DBG:
        print("[Step 4] shading complete\n")

    return np.clip(img, 0, 1)
