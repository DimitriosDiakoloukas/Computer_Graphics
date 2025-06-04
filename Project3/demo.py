import numpy as np
import os
import imageio.v2 as imageio  # Suppress imread warning
from all_funcs_new import render_object
from MatPhong import MatPhong

def main():
    # Load data
    data = np.load('hw3.npy', allow_pickle=True).item()

    # Extract data
    v_pos = data['v_pos']
    v_uvs = data['v_uvs'].T
    t_pos_idx = data['t_pos_idx'].T
    cam_pos = data['cam_pos'].flatten()
    up = data['up'].flatten()
    target = data['target'].flatten()
    l_pos = [np.array(l).flatten() for l in data['l_pos']]
    l_int = [np.array(l) for l in data['l_int']]
    l_amb = data['l_amb']
    plane_h = data['plane_h']
    plane_w = data['plane_w']
    res_h = data['res_h']
    res_w = data['res_w']
    focal = data['focal']

    ka_base = data['ka']
    kd_base = data['kd']
    ks_base = data['ks']
    n_base = data['n']

    # Load texture
    texture_path = "Mona-Lisa-Exist-in-Real-Life-2635825581.jpg"
    if os.path.exists(texture_path):
        tex = imageio.imread(texture_path) / 255.0
        print(f"Loaded texture: {texture_path}, shape: {tex.shape}")
    else:
        print(f"Texture file '{texture_path}' not found. Using white texture.")
        tex = np.ones((512, 512, 3), dtype=np.float32)

    print(f"Texture range: min={tex.min()}, max={tex.max()}, mean={tex.mean()}")
    print(f"UVs: min={v_uvs.min()}, max={v_uvs.max()}")

    os.makedirs("results", exist_ok=True)

    # Render base 8 images (4 light modes × 2 shaders)
    shaders = ['gouraud', 'phong']
    modes = {
        'ambient': (ka_base, 0.0, 0.0),
        'diffuse': (0.0, kd_base, 0.0),
        'specular': (0.0, 0.0, ks_base),
        'all': (ka_base, kd_base, ks_base)
    }

    for shader in shaders:
        for mode_name, (ka, kd, ks) in modes.items():
            mat = MatPhong(ka=ka, kd=kd, ks=ks, n=n_base)

            img = render_object(
                v_pos=v_pos,
                v_uvs=v_uvs,
                t_pos_idx=t_pos_idx,
                tex=tex,
                plane_h=plane_h,
                plane_w=plane_w,
                res_h=res_h,
                res_w=res_w,
                focal=focal,
                eye=cam_pos,
                target=target,
                up=up,
                mat=mat,
                l_pos=l_pos,
                l_int=l_int,
                l_amb=l_amb,
                shader=shader
            )

            print(f"[{shader}-{mode_name}] img stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

            filename = f"results/{shader}_{mode_name}.png"
            imageio.imwrite(filename, (img * 255).astype(np.uint8))
            print(f"Saved: {filename}")

    # Phong-only: individual light sources
    mat_all = MatPhong(ka_base, kd_base, ks_base, n_base)

    for i in range(3):
        single_l_pos = [l_pos[i]]
        single_l_int = [l_int[i]]

        img = render_object(
            v_pos=v_pos,
            v_uvs=v_uvs,
            t_pos_idx=t_pos_idx,
            tex=tex,
            plane_h=plane_h,
            plane_w=plane_w,
            res_h=res_h,
            res_w=res_w,
            focal=focal,
            eye=cam_pos,
            target=target,
            up=up,
            mat=mat_all,
            l_pos=single_l_pos,
            l_int=single_l_int,
            l_amb=l_amb,
            shader='phong'
        )

        print(f"[phong-light-{i+1}] img stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

        filename = f"results/phong_light_{i+1}.png"
        imageio.imwrite(filename, (img * 255).astype(np.uint8))
        print(f"Saved: {filename}")

    # All lights again
    img = render_object(
        v_pos=v_pos,
        v_uvs=v_uvs,
        t_pos_idx=t_pos_idx,
        tex=tex,
        plane_h=plane_h,
        plane_w=plane_w,
        res_h=res_h,
        res_w=res_w,
        focal=focal,
        eye=cam_pos,
        target=target,
        up=up,
        mat=mat_all,
        l_pos=l_pos,
        l_int=l_int,
        l_amb=l_amb,
        shader='phong'
    )

    print(f"[phong-light-all] img stats: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

    filename = "results/phong_light_all.png"
    imageio.imwrite(filename, (img * 255).astype(np.uint8))
    print(f"Saved: {filename}")

if __name__ == "__main__":
    main()
