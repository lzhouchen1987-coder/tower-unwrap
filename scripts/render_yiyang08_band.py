# -*- coding: utf-8 -*-
"""宜阳8#塔：拟合剖面 + 渲染一段 2mm/px(ss=2) 高清展开带，与成品对比"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_yiyang_08"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"
PROF = os.path.join(OUT, "yiyang08_profile.npy")


def main():
    from core.unwrap_core import load_blocks, fit_axis, AxisProfile, build_raycast_scene, render_unwrap
    t0 = time.time()
    print("1) 加载模型分块...", flush=True)
    blocks = load_blocks(ROOT)
    print(f"   {len(blocks)}块, {time.time()-t0:.1f}s", flush=True)

    print("2) 拟合塔轴...", flush=True)
    prof = fit_axis(blocks)
    profile = AxisProfile(prof)
    np.save(PROF, prof)
    print(f"   z范围 {profile.z_min:.1f}~{profile.z_max:.1f}m, r_max {profile.r_max:.2f}m", flush=True)
    for row in prof:
        print(f"   z={row[0]:7.1f}  cx={row[1]:8.3f} cy={row[2]:8.3f} r={row[3]:6.3f}", flush=True)

    print("3) 构建光线投射场景...", flush=True)
    scene_data = build_raycast_scene(blocks)

    # 高清展开带：混凝土段中部 10m，2mm/px，2倍超采样
    zmid0 = profile.z_min + (profile.z_max - profile.z_min) * 0.35
    zmid1 = zmid0 + 10.0
    t0 = time.time()
    print(f"4) 渲染高清带 z=[{zmid0:.1f},{zmid1:.1f}] 2mm/px ss=2...", flush=True)
    def prog(i, n):
        print(f"   tile {i}/{n}  {time.time()-t0:.0f}s", flush=True)
    img, meta = render_unwrap(blocks, scene_data, profile,
                              z_bottom=zmid0, z_top=zmid1,
                              px_m=0.002, theta0=0.0, row_tile=1024, ss=2, progress=prog)
    out = os.path.join(OUT, "yiyang08_band_2mm.png")
    img.save(out)
    print(f"   尺寸 {meta['W']}x{meta['H']} 已保存 {out}  渲染耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
