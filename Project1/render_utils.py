import numpy as np
import cv2
from flat_shading import f_shading
from texture_shading import t_shading 

def load_png_image(filepath):
    """
    Load a PNG image and convert it to RGB with values in [0, 1].
    """
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or path is incorrect: {filepath}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return img_rgb


def render_img(faces, vertices, vcolors, uvs, depth, shading, textImg=None):
    """
    Renders a 3D object's projected triangles into a 2D image using flat or texture shading.

    Parameters:
        faces (Kx3 ndarray): Each row defines a triangle (3 vertex indices)
        vertices (Lx2 ndarray): 2D projected coordinates for each vertex
        vcolors (Lx3 ndarray): RGB color per vertex (for flat shading)
        depth (Lx1 ndarray or L,) : Depth value per vertex
        shading (str): "f" (flat) or "t" (texture)
        uv (Lx2 ndarray): UV coordinates per vertex (for texture shading)
        textImg (HxWx3 ndarray): Texture image in RGB format, values in [0, 1]

    Returns:
        img (MxNx3 ndarray): Final rendered image
    """
    M, N = 512, 512
    img = np.ones((M, N, 3), dtype=np.float32)  # white background

    # Ensure depth is 1D
    depth = depth.flatten()

    # Compute average depth of each triangle for back-to-front sorting
    triangle_depths = np.mean(depth[faces], axis=1)

    # Sort triangles by depth descending (farthest first)
    sorted_indices = np.argsort(-triangle_depths)
    cnt = 0
    for i in sorted_indices:
        face = faces[i]
        triangle_vertices = vertices[face, :]

        if shading == "f":
            triangle_colors = vcolors[face, :]
            img = f_shading(img, triangle_vertices, triangle_colors)

        elif shading == "t":
            if uvs is None or textImg is None:
                raise ValueError("Texture shading requires both uv and textImg.")
            triangle_uv = uvs[face, :]
            img = t_shading(img, triangle_vertices, triangle_uv, textImg)

        else:
            raise ValueError("Invalid shading mode. Use 'f' or 't'.")
        cnt += 1
        # Optional: Debug print
        print(f"Rendering triangle {i} (face: {face.tolist()})")
        print(f"COUNTER: {cnt}")
    print(f"Total triangles: {len(sorted_indices)}")
    return img
