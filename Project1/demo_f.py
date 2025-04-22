import numpy as np
import cv2
from render_utils import render_img

def main():
    data = np.load("hw1.npy", allow_pickle=True).item()

    vertices = data["v_pos2d"]
    vcolors = data["v_clr"]
    faces = np.array(data["t_pos_idx"])
    depth = data["depth"].reshape(-1, 1)
    uvs = data["v_uvs"]

    # FIX: Normalize and scale vertices to fit 512x512 image if you want perfect fit to canvas
    # Normalize to range [0, 1]
    # vertices = vertices - np.min(vertices, axis=0)
    # vertices = vertices / np.max(vertices, axis=0)

    # # Scale to canvas size
    # vertices = vertices * 511  # 0 to 511

    img = render_img(faces, vertices, vcolors, uvs, depth, shading="f", textImg=None)

    img_out = (img * 255).astype(np.uint8)
    cv2.imwrite("flat_result.png", img_out)

if __name__ == "__main__":
    main()
