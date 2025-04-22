import numpy as np
import cv2
from render_utils import render_img, load_png_image

def main():
    data = np.load("hw1.npy", allow_pickle=True).item()

    vertices = data["v_pos2d"]
    vcolors = data["v_clr"]
    faces = np.array(data["t_pos_idx"])
    depth = data["depth"].reshape(-1, 1)
    uvs = data["v_uvs"]

    texture = load_png_image("texImg.jpg")

    # Normalize vertices to fit 512x512 canvas if you want perfect fit in canvas
    # vertices = vertices - np.min(vertices, axis=0)
    # vertices = vertices / np.max(vertices, axis=0)
    # vertices = vertices * 511

    img = render_img(faces, vertices, vcolors, uvs, depth, shading="t", texImg=texture)
    print("Image shape:", img.shape)      # (height, width, 3)
    print("Image dtype:", img.dtype)      # float32 (usually before saving)
    print("Image min/max values:", img.min(), img.max())  # to check if normalized

    img_out = (img * 255).astype(np.uint8)
    print("Final image info:")
    print("Shape:", img_out.shape)
    print("Dtype:", img_out.dtype)
    print("Min/Max:", img_out.min(), img_out.max())

    cv2.imwrite("texture_result.png", img_out)

if __name__ == "__main__":
    main()