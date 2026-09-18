# -*- coding: utf-8 -*-
"""诊断五河S02顶部段：双重表面占比 + 两层三角形UV密度对比"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_wuhe_s02"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
Z0, Z1 = 56.0, 70.0
PX_M = 0.01


def main():
    import open3d as o3d
    from core.unwrap_core import load_blocks, fit_axis, AxisProfile, build_raycast_scene
    blocks = load_blocks(ROOT)
    prof = fit_axis(blocks)
    profile = AxisProfile(prof)
    print(f"profile z {profile.z_min:.0f}~{profile.z_max:.0f}", flush=True)
    scene, T, UV, TEX, tex_registry = build_raycast_scene(blocks)
    Vg = np.vstack([b["vertices"] for b in blocks])

    # 每个三角形的UV密度：uv面积*纹理像素数/世界面积
    def tri_scores():
        uv0 = UV[0::3]; uv1 = UV[1::3]; uv2 = UV[2::3]
        uv_area = 0.5 * np.abs((uv1[:, 0] - uv0[:, 0]) * (uv2[:, 1] - uv0[:, 1])
                               - (uv2[:, 0] - uv0[:, 0]) * (uv1[:, 1] - uv0[:, 1]))
        v0 = Vg[T[:, 0]]; v1 = Vg[T[:, 1]]; v2 = Vg[T[:, 2]]
        w_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) + 1e-12
        tex_px = np.ones(len(T))
        for gid in np.unique(TEX):
            e = tex_registry[gid]
            if e:
                tex_px[TEX == gid] = e[1][0] * e[1][1]
        return uv_area * tex_px / w_area

    scores = tri_scores()
    print(f"密度分位: 10%={np.percentile(scores,10):.2e} 50%={np.percentile(scores,50):.2e} 90%={np.percentile(scores,90):.2e}", flush=True)

    # 网格光线
    s1v = float(profile.slant(np.array([Z1]))[0])
    _, _, rr_rng = profile.eval(np.linspace(Z0, Z1, 32))
    W = int(2 * np.pi * rr_rng.max() / PX_M)
    H = int((Z1 - Z0) / PX_M)
    rays = []
    for py in range(0, H, 4):
        vv = s1v - (py + 0.5) * PX_M
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        cx, cy, rr = profile.eval(np.array([zz]))
        cx, cy, rr = float(cx[0]), float(cy[0]), float(rr[0])
        for px in range(0, W, 4):
            uu = (px + 0.5 - W / 2) * PX_M
            if abs(uu) > np.pi * rr:
                continue
            th = uu / rr
            rays.append([cx + np.cos(th) * (rr + 1.5), cy + np.sin(th) * (rr + 1.5), zz,
                         -np.cos(th), -np.sin(th), 0.0])
    rays = np.array(rays, np.float32)
    print(f"光线 {len(rays)}", flush=True)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    prim1 = ans["primitive_ids"].numpy()
    t1 = ans["t_hit"].numpy()
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID
    ok1 = (prim1 != INVALID) & (t1 < 2.5)
    # 第二击
    adv = rays.copy()
    adv[:, :3] += rays[:, 3:] * (t1[:, None] + 0.03)
    ans2 = scene.cast_rays(o3d.core.Tensor(adv[ok1]))
    prim2 = ans2["primitive_ids"].numpy()
    t2 = ans2["t_hit"].numpy()
    close = (prim2 != INVALID) & (t2 < 0.15)
    print(f"首击有效 {ok1.mean()*100:.1f}%，其中 15cm 内有第二层 {close.mean()*100:.1f}%", flush=True)
    p1 = prim1[ok1][close]
    p2 = prim2[close]
    s1 = scores[p1]; s2 = scores[p2]
    better2 = s2 > s1 * 1.3
    print(f"双层处：第二层密度显著更高 {better2.mean()*100:.1f}%，第一层更高 {((s1 > s2*1.3).mean())*100:.1f}%", flush=True)
    print(f"双层处首击密度中位 {np.median(s1):.2e} vs 第二击 {np.median(s2):.2e}", flush=True)


if __name__ == "__main__":
    main()
