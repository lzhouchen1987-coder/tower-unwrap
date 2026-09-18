# -*- coding: utf-8 -*-
"""验证：黑沟像素沿光线深入后，第二层表面是否有有效纹理"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_yiyang_08"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
PROF = os.path.join(OUT, "yiyang08_profile.npy")
Z0, Z1 = 1.6, 11.6
PX_M = 0.002
PIXELS = [(6112, 3529), (5822, 3949), (6400, 3600), (5252, 3409), (6052, 4609)]


def main():
    import open3d as o3d
    from core.unwrap_core import load_blocks, AxisProfile, build_raycast_scene, MaskPool
    blocks = load_blocks(ROOT)
    profile = AxisProfile(np.load(PROF))
    scene, T, UV, TEX, tex_registry = build_raycast_scene(blocks)
    mask_pool = MaskPool(tex_registry)
    W = 9975
    s1 = float(profile.slant(np.array([Z1]))[0])
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID

    def uv_valid(gid, uv):
        vm = mask_pool.get(gid)
        if vm is None:
            return None
        mh, mw = vm.shape
        mx = int(np.clip((uv[0] % 1.0) * mw, 0, mw - 1))
        my = int(np.clip((uv[1] % 1.0) * mh, 0, mh - 1))
        return bool(vm[my, mx])

    for px, py in PIXELS:
        vv = s1 - (py + 0.5) * PX_M
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        cx, cy, rr = profile.eval(np.array([zz]))
        cx, cy, rr = float(cx[0]), float(cy[0]), float(rr[0])
        uu = (px + 0.5 - W / 2) * PX_M
        th = uu / rr
        org = np.array([cx + np.cos(th) * (rr + 1.5), cy + np.sin(th) * (rr + 1.5), zz])
        d = np.array([-np.cos(th), -np.sin(th), 0.0])
        print(f"\n像素({px},{py}) z={zz:.2f} theta={np.degrees(th):.1f}°", flush=True)
        t_adv = 0.0
        for k in range(5):
            o = org + d * t_adv
            ray = np.concatenate([o, d]).astype(np.float32)[None]
            ans = scene.cast_rays(o3d.core.Tensor(ray))
            prim = int(ans["primitive_ids"].numpy()[0])
            if prim == INVALID:
                print(f"  第{k+1}层: 未命中", flush=True)
                break
            t = float(ans["t_hit"].numpy()[0])
            bary = ans["primitive_uvs"].numpy()[0]
            w1, w2 = float(bary[0]), float(bary[1])
            uv = UV[prim * 3] * (1 - w1 - w2) + UV[prim * 3 + 1] * w1 + UV[prim * 3 + 2] * w2
            gid = int(TEX[prim])
            hit_r = rr + 1.5 - (t_adv + t)
            print(f"  第{k+1}层: t+={t:.3f} 半径={hit_r:.3f} gid={gid} uv有效={uv_valid(gid, uv)}", flush=True)
            t_adv += t + 0.015


if __name__ == "__main__":
    main()
