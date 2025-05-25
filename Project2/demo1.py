import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from all_funcs import render_object

def load_data():
    """
    Load the scene data and texture image from files.
    Returns:
        data (dict): Dictionary containing scene data including vertex positions, UVs, and camera parameters.
        texture (np.ndarray): Texture image for the object.
    """
    data = np.load("hw2.npy", allow_pickle=True).item()
    texture = np.asarray(Image.open("stone-72_diffuse.jpg")) / 255.0
    return data, texture


def render_forward_demo(data, texture, output_dir="demo_1", video_name="demo_1_video.mp4"):
    """
    Render a forward demo of a car moving along a circular path with a camera tracking it.
    Args:
        data (dict): Dictionary containing scene data including vertex positions, UVs, and camera parameters.
        texture (np.ndarray): Texture image for the object.
        output_dir (str): Directory to save the rendered frames.
        video_name (str): Name of the output video file.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Unpack scene data
    v_pos = data["v_pos"].T
    v_uvs = data["v_uvs"]
    t_pos_idx = np.array(data["t_pos_idx"])
    v_clr = np.ones_like(v_pos)

    center = data["k_road_center"]
    radius = data["k_road_radius"]
    speed = data["car_velocity"]
    cam_offset = data["k_cam_car_rel_pos"]
    duration = data["k_duration"]
    fps = data["k_fps"]
    up = data["k_cam_up"].flatten()
    focal = data["k_f"]
    plane_h = data["k_sensor_height"]
    plane_w = data["k_sensor_width"]
    res_h = res_w = 512

    total_frames = duration * fps
    omega = speed / radius

    print(f"Rendering {total_frames} frames...")

    for frame in range(total_frames):
        t = frame / fps
        theta = omega * t

        car_pos = center + radius * np.array([np.cos(theta), 0, np.sin(theta)])
        tangent = np.array([-np.sin(theta), 0, np.cos(theta)])
        tangent /= np.linalg.norm(tangent)

        cam_pos = car_pos + cam_offset
        target = cam_pos + tangent

        img = render_object(
            v_pos=v_pos,
            v_clr=v_clr,
            t_pos_idx=t_pos_idx,
            plane_h=plane_h,
            plane_w=plane_w,
            res_h=res_h,
            res_w=res_w,
            focal=focal,
            eye=cam_pos,
            up=up,
            target=target,
            v_uvs=v_uvs,
            texImg=texture,
        )

        plt.imsave(f"{output_dir}/frame_{frame:03d}.png", img)
        print(f"Saved frame {frame + 1}/{total_frames}", end="\r")

    print("\nAll frames saved.")
    save_video(output_dir, video_name, fps)

def save_video(folder, output_name, fps):
    """
    Save the rendered frames as a video file.
    Args:
        folder (str): Directory containing the rendered frames.
        output_name (str): Name of the output video file.
        fps (int): Frames per second for the video.
    """
    frames = []
    for i in range(fps * 5):  # 5 seconds at 25fps = 125 frames
        filename = os.path.join(folder, f"frame_{i:03d}.png")
        image = imageio.imread(filename)
        frames.append(image)

    output_path = f"{output_name}"
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved as {output_path}")


if __name__ == "__main__":
    data, texImg = load_data()
    render_forward_demo(data, texImg)
