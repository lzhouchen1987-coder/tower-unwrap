# -*- coding: utf-8 -*-
"""诊断五河S02顶部段青色描边：渲染RAW小带→找青色像素→溯源图集"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_wuhe_s02"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
Z0, Z1 = 62.0, 68.0
PX_M = 0.005
BAND = os.path.join(OUT, "wuhe_top_raw.png")


def main():
    import open3d as o3d
    import cv2
    from core.unwrap_core import (load_blocks, fit_axis, AxisProfile, build_raycast_scene,
                                  render_unwrap, TexturePool, _parse_mtl_textures)
    blocks = load_blocks(ROOT)
    prof = fit_axis(blocks)
    profile = AxisProfile(prof)
    scene_data = build_raycast_scene(blocks)
    scene, T, UV, TEX, tex_registry = scene_data
    if os.path.exists(BAND):
        img = np.asarray(Image.open(BAND))
        meta = {"W": img.shape[1], "H": img.shape[0]}
    else:
        img_pil, meta = render_unwrap(blocks, scene_data, profile, z_bottom=Z0, z_top=Z1,
                                      px_m=PX_M, theta0=0.0, row_tile=1024, ss=1)
        img_pil.save(BAND)
        img = np.asarray(img_pil)
    W, H = meta["W"], meta["H"]
    rgb, a = img[..., :3], img[..., 3]
    d = rgb[..., 2].astype(np.int16) - rgb[..., 0].astype(np.int16)
    cyan = ((d > 35) & (rgb[..., 2] > 130) & (a > 200)).astype(np.uint8)
    print(f"画布 {W}x{H} 青色像素占比 {cyan.mean()*100:.2f}%", flush=True)
    small = cv2.resize(cyan, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    n, lab, stats, cent = cv2.connectedComponentsWithStats((small > 0.3).astype(np.uint8), 8)
    order = np.argsort(-stats[1:, 4])[:6] + 1
    samples = []
    for k in order:
        cx2, cy2 = cent[k]
        x0 = int(np.clip(cx2 * 2 - 20, 0, W - 40)); y0 = int(np.clip(cy2 * 2 - 20, 0, H - 40))
        win = d[y0:y0 + 40, x0:x0 + 40].copy()
        win[cyan[y0:y0 + 40, x0:x0 + 40] == 0] = -999
        iy, ix = np.unravel_index(win.argmax(), win.shape)
        samples.append((x0 + ix, y0 + iy))

    gid_map, tex_map = {}, {}
    for bi, b in enumerate(blocks):
        for li in np.unique(b["tri_mat"]):
            key = (bi, int(li))
            if key not in tex_map:
                tex_map[key] = len(gid_map)
                gid_map[len(gid_map)] = key
    pool = TexturePool(tex_registry)
    s1v = float(profile.slant(np.array([Z1]))[0])
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID
    for px, py in samples:
        vv = s1v - (py + 0.5) * PX_M
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        cx, cy, rr = profile.eval(np.array([zz]))
        cx, cy, rr = float(cx[0]), float(cy[0]), float(rr[0])
        th = ((px + 0.5 - W / 2) * PX_M) / rr
        ray = np.array([[cx + np.cos(th) * (rr + 1.5), cy + np.sin(th) * (rr + 1.5), zz,
                         -np.cos(th), -np.sin(th), 0.0]], np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(ray))
        prim = int(ans["primitive_ids"].numpy()[0])
        print(f"\n像素({px},{py}) RGB={rgb[py,px].tolist()} B-R={int(d[py,px])} z={zz:.2f}", flush=True)
        if prim == INVALID:
            print("  未命中", flush=True)
            continue
        bary = ans["primitive_uvs"].numpy()[0]
        w1, w2 = float(bary[0]), float(bary[1])
        uv = UV[prim * 3] * (1 - w1 - w2) + UV[prim * 3 + 1] * w1 + UV[prim * 3 + 2] * w2
        gid = int(TEX[prim])
        bi, li = gid_map[gid]
        texmap = _parse_mtl_textures(blocks[bi]["path"], fix_gutters=False)
        ofile = texmap[li][0]
        with Image.open(ofile) as oim:
            full = oim.size
        # OBJ v翻转后采样（与渲染一致）
        fx = int((uv[0] % 1.0) * (full[0] - 1)); fy = int((1.0 - uv[1] % 1.0) * (full[1] - 1))
        with Image.open(ofile) as oim:
            crop = np.asarray(oim.convert("RGB").crop((max(fx - 60, 0), max(fy - 60, 0), fx + 60, fy + 60)))
        print(f"  图集={os.path.basename(ofile)} 落点({fx},{fy}) 周边均值RGB={crop.mean((0,1)).round(0).tolist()}", flush=True)


if __name__ == "__main__":
    main()
