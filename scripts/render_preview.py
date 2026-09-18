# -*- coding: utf-8 -*-
"""验证：五河S02 全塔低清展开预览（2cm/px）"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from core.unwrap_core import load_blocks, fit_axis, AxisProfile, build_raycast_scene, render_unwrap

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_wuhe_s02"
OUT = r"D:\KimiData\kimi\workspace\tower-unwrap\output"

t0 = time.time()
print("1) 加载模型分块...")
blocks = load_blocks(ROOT)
print(f"   {len(blocks)}块, {time.time()-t0:.1f}s")

t0 = time.time()
print("2) 拟合塔轴...")
prof = fit_axis(blocks)
profile = AxisProfile(prof)
print(f"   z范围 {profile.z_min:.1f}~{profile.z_max:.1f}m, r_max {profile.r_max:.2f}m, {time.time()-t0:.1f}s")

t0 = time.time()
print("3) 构建光线投射场景...")
scene_data = build_raycast_scene(blocks)
print(f"   {time.time()-t0:.1f}s")

t0 = time.time()
print("4) 渲染展开图 (2cm/px)...")
def prog(i, n):
    if i % 2 == 0 or i == n:
        print(f"   tile {i}/{n}  {time.time()-t0:.0f}s")
img, meta = render_unwrap(blocks, scene_data, profile,
                          z_bottom=-62.5, z_top=70.0,
                          px_m=0.02, theta0=0.0, row_tile=1024, progress=prog)
print(f"   尺寸 {meta['W']}x{meta['H']}  宽{meta['width_m']:.1f}m 高{meta['height_m']:.1f}m")
out = os.path.join(OUT, "wuhe_s02_preview_2cm.png")
img.save(out)
print(f"   已保存 {out}  总耗时 {time.time()-t0:.0f}s")
