# -*- coding: utf-8 -*-
"""溯源：找出展开带中蓝色补丁像素来自哪张图集、增益是否生效、原始颜色是什么"""
import sys, os, time, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_yiyang_08"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
PROF = os.path.join(OUT, "yiyang08_profile.npy")
BAND = os.path.join(OUT, "yiyang08_band_2mm.png")
Z0, Z1 = 1.6, 11.6
PX_M, SS = 0.002, 2


def main():
    import cv2
    from core.unwrap_core import (load_blocks, AxisProfile, build_raycast_scene,
                                  TexturePool, _parse_mtl_textures)
    t0 = time.time()
    print("加载模型...", flush=True)
    blocks = load_blocks(ROOT)
    prof = np.load(PROF)
    profile = AxisProfile(prof)
    scene, T, UV, TEX, tex_registry = build_raycast_scene(blocks)
    Vg = np.vstack([b["vertices"] for b in blocks])
    print(f"完成 {time.time()-t0:.0f}s, 纹理表 {len(tex_registry)} 项", flush=True)

    # 复刻 render_unwrap 的 gid -> (block, local_idx) 映射
    gid_map, tex_map = {}, {}
    for bi, b in enumerate(blocks):
        for li in np.unique(b["tri_mat"]):
            key = (bi, int(li))
            if key not in tex_map:
                tex_map[key] = len(gid_map)
                gid_map[len(gid_map)] = key

    # 复刻跨图集增益
    meds = np.full((len(tex_registry), 3), np.nan)
    for i, e in enumerate(tex_registry):
        if e and e[2]:
            meds[i] = np.asarray(e[2], float)[:3]
    global_med = np.nanmedian(meds, axis=0)
    safe = np.where(np.isfinite(meds), meds, global_med[None, :])
    gains = np.clip(global_med[None, :] / safe, 0.5, 2.0)
    gains[~np.isfinite(meds)] = 1.0
    print(f"全模型中位色 RGB={global_med.round(1)}", flush=True)

    # 支持命令行传入指定像素 "px,py px,py ..."，否则自动扫描蓝色补丁
    cli_pixels = []
    for a in sys.argv[1:]:
        if "," in a:
            cli_pixels.append(tuple(int(v) for v in a.split(",")) + (0,))
    if cli_pixels:
        samples = cli_pixels
        pim = Image.open(BAND)
        W, H = pim.size
        img = np.asarray(pim)
        d = img[..., 2].astype(np.int16) - img[..., 0].astype(np.int16)
    else:
        # 在展开带里找蓝色补丁（B-R 大）
        print("扫描蓝色补丁...", flush=True)
        pim = Image.open(BAND)
        W, H = pim.size
        img = np.asarray(pim)                       # RGBA
        d = img[..., 2].astype(np.int16) - img[..., 0].astype(np.int16)
        mask = ((d > 18) & (img[..., 3] > 200)).astype(np.uint8)
        small = cv2.resize(mask, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        n, lab, stats, cent = cv2.connectedComponentsWithStats((small > 0.25).astype(np.uint8), 8)
        order = np.argsort(-stats[1:, 4])[:8] + 1
        samples = []
        for k in order:
            cx4, cy4 = cent[k]
            # 在全分辨率簇心附近取 d 最大的像素
            x0 = int(np.clip(cx4 * 4 - 30, 0, W - 60)); y0 = int(np.clip(cy4 * 4 - 30, 0, H - 60))
            win = d[y0:y0 + 60, x0:x0 + 60].copy()
            win[mask[y0:y0 + 60, x0:x0 + 60] == 0] = -999
            iy, ix = np.unravel_index(win.argmax(), win.shape)
            samples.append((x0 + ix, y0 + iy, stats[k, 4] * 16))
    print(f"取 {len(samples)} 个样本点", flush=True)

    # 逐样本反查
    s1 = float(profile.slant(np.array([Z1]))[0])
    pool = TexturePool(tex_registry)
    INVALID = 4294967295
    for px, py, area in samples:
        vv = s1 - (py + 0.5) * PX_M
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        cx, cy, rr = profile.eval(np.array([zz]))
        cx, cy, rr = float(cx[0]), float(cy[0]), float(rr[0])
        uu = (px + 0.5 - W / 2) * PX_M
        th = uu / rr
        # 外侧投射（与 render 一致）
        ray = np.array([[cx + np.cos(th) * (rr + 1.5), cy + np.sin(th) * (rr + 1.5), zz,
                         -np.cos(th), -np.sin(th), 0.0]], np.float32)
        ans = scene.cast_rays(__import__("open3d").core.Tensor(ray))
        prim = int(ans["primitive_ids"].numpy()[0])
        t_hit = float(ans["t_hit"].numpy()[0])
        out_rgb = img[py, px, :3]
        print(f"\n像素({px},{py}) 输出RGB={out_rgb.tolist()} B-R={int(d[py,px])} 簇面积约{area}px", flush=True)
        print(f"  z={zz:.2f}m theta={np.degrees(th):.1f}° r={rr:.3f}", flush=True)
        if prim == INVALID or t_hit > 1.5 + 1.0:
            print(f"  外侧未命中(t={t_hit:.2f})，该像素来自内侧补投或空洞", flush=True)
            continue
        bary = ans["primitive_uvs"].numpy()[0]
        w1, w2 = float(bary[0]), float(bary[1])
        uv = UV[prim * 3] * (1 - w1 - w2) + UV[prim * 3 + 1] * w1 + UV[prim * 3 + 2] * w2
        gid = int(TEX[prim])
        bi, li = gid_map[gid]
        entry = tex_registry[gid]
        cache_path = os.path.basename(entry[0]) if entry else "None"
        tri_c = Vg[T[prim]]
        zmin_t, zmax_t = float(tri_c[:, 2].min()), float(tri_c[:, 2].max())
        print(f"  命中prim={prim} gid={gid} block={os.path.basename(blocks[bi]['path'])} mtl={li}", flush=True)
        print(f"  三角形z范围 [{zmin_t:.2f},{zmax_t:.2f}] 缓存={cache_path} med={entry[2] if entry else None}", flush=True)
        print(f"  增益={gains[gid].round(3).tolist()}", flush=True)
        # 缓存图采样色 + 周边黑沟/楔状区分析
        cim = pool.get(gid)
        h, w = cim.shape[:2]
        x = int((uv[0] % 1.0) * (w - 1)); y = int((uv[1] % 1.0) * (h - 1))
        c_raw = cim[y, x]
        print(f"  缓存图采样RGB={c_raw.tolist()} ×增益={(c_raw.astype(float)*gains[gid]).round(0).tolist()}", flush=True)
        from scipy.ndimage import distance_transform_edt
        wx0, wy0 = max(x - 100, 0), max(y - 100, 0)
        win = cim[wy0:min(y + 101, h), wx0:min(x + 101, w)]
        dark = win.sum(-1) <= 105
        dist = distance_transform_edt(~dark)
        print(f"  采样点在图集({x},{y}) 距最近黑沟={dist[y-wy0, x-wx0]:.1f}px 窗口黑占比={dark.mean():.2f}", flush=True)
        # 找回原始图集路径并采样
        texmap = _parse_mtl_textures(blocks[bi]["path"], fix_gutters=False)
        orig = texmap.get(li)
        if orig:
            ofile = orig[0]
            with Image.open(ofile) as oim:
                oim.thumbnail((2048, 2048))   # 降采样加速读全图
                osmall = np.asarray(oim.convert("RGB"))
            oh, ow = osmall.shape[:2]
            with Image.open(ofile) as oim2:
                full = oim2.size
            fx = int((uv[0] % 1.0) * (full[0] - 1)); fy = int((uv[1] % 1.0) * (full[1] - 1))
            with Image.open(ofile) as oim3:
                crop = np.asarray(oim3.convert("RGB").crop((max(fx - 8, 0), max(fy - 8, 0), fx + 8, fy + 8)))
            print(f"  原始图集={os.path.basename(ofile)} 该处RGB={crop.mean((0,1)).round(0).tolist()}", flush=True)


if __name__ == "__main__":
    main()
