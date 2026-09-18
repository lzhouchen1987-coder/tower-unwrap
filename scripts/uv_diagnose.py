# -*- coding: utf-8 -*-
"""诊断：把判废像素的UV落点画到对应图集缩略图上，看是楔边带还是大片虚空"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_yiyang_08"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
PROF = os.path.join(OUT, "yiyang08_profile.npy")
Z0, Z1 = 1.6, 11.6
PX_M = 0.002
STEP = 12   # 每12个像素抽一根光线


def main():
    import open3d as o3d
    from core.unwrap_core import load_blocks, AxisProfile, build_raycast_scene, MaskPool, _parse_mtl_textures
    blocks = load_blocks(ROOT)
    profile = AxisProfile(np.load(PROF))
    scene, T, UV, TEX, tex_registry = build_raycast_scene(blocks)
    mask_pool = MaskPool(tex_registry)
    # gid -> (block, local) 与 原始图集路径
    gid_map, tex_map = {}, {}
    for bi, b in enumerate(blocks):
        for li in np.unique(b["tri_mat"]):
            key = (bi, int(li))
            if key not in tex_map:
                tex_map[key] = len(gid_map)
                gid_map[len(gid_map)] = key
    W, H = 9975, 5001
    s1 = float(profile.slant(np.array([Z1]))[0])
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID

    rays, coords = [], []
    for py in range(0, H, STEP):
        vv = s1 - (py + 0.5) * PX_M
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        cx, cy, rr = profile.eval(np.array([zz]))
        cx, cy, rr = float(cx[0]), float(cy[0]), float(rr[0])
        for px in range(0, W, STEP):
            uu = (px + 0.5 - W / 2) * PX_M
            if abs(uu) > np.pi * rr:
                continue
            th = uu / rr
            rays.append([cx + np.cos(th) * (rr + 1.5), cy + np.sin(th) * (rr + 1.5), zz,
                         -np.cos(th), -np.sin(th), 0.0])
            coords.append((px, py))
    rays = np.array(rays, np.float32)
    print(f"光线 {len(rays)} 根", flush=True)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    prim = ans["primitive_ids"].numpy()
    bary = ans["primitive_uvs"].numpy()
    t_hit = ans["t_hit"].numpy()
    bad = defaultdict(list)   # gid -> [uv,...]
    n_hit = n_bad = n_flip_ok = 0
    for i in range(len(rays)):
        p = int(prim[i])
        if p == INVALID or t_hit[i] > 2.5:
            continue
        n_hit += 1
        w1, w2 = float(bary[i][0]), float(bary[i][1])
        uv = UV[p * 3] * (1 - w1 - w2) + UV[p * 3 + 1] * w1 + UV[p * 3 + 2] * w2
        gid = int(TEX[p])
        vm = mask_pool.get(gid)
        if vm is None:
            continue
        mh, mw = vm.shape
        mx = int(np.clip((uv[0] % 1.0) * mw, 0, mw - 1))
        my = int(np.clip((uv[1] % 1.0) * mh, 0, mh - 1))
        if not vm[my, mx]:
            bad[gid].append((uv[0] % 1.0, uv[1] % 1.0))
            n_bad += 1
        if vm[1023 - my, mx]:
            n_flip_ok += 1
    print(f"命中 {n_hit}，判废 {n_bad}（{n_bad/max(n_hit,1)*100:.1f}%），涉及图集 {len(bad)} 张", flush=True)
    print(f"v不翻转有效率 {(n_hit-n_bad)/max(n_hit,1)*100:.1f}%  vs  v翻转有效率 {n_flip_ok/max(n_hit,1)*100:.1f}%", flush=True)

    top = Counter({g: len(v) for g, v in bad.items()}).most_common(4)
    for gid, cnt in top:
        bi, li = gid_map[gid]
        texmap = _parse_mtl_textures(blocks[bi]["path"], fix_gutters=False)
        ofile = texmap[li][0]
        with Image.open(ofile) as im:
            full = im.size
            im.thumbnail((1200, 1200))
            thumb = np.asarray(im.convert("RGB")).copy()
        th_h, th_w = thumb.shape[:2]
        for u, v in bad[gid]:
            x = int(u * (th_w - 1)); y = int(v * (th_h - 1))
            thumb[max(y-2,0):y+3, max(x-2,0):x+3] = [255, 0, 0]
        out = os.path.join(OUT, f"uv_diag_gid{gid}.jpg")
        Image.fromarray(thumb).save(out, quality=88)
        print(f"gid={gid} 判废{cnt}点 -> {os.path.basename(ofile)} 已保存 {out}", flush=True)


if __name__ == "__main__":
    main()
