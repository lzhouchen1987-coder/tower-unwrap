# -*- coding: utf-8 -*-
"""诊断：取展开图中一小片区域，检查黑斑像素的 UV/纹理采样是否正确"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import open3d as o3d
from core.unwrap_core import (load_blocks, fit_axis, AxisProfile,
                              build_raycast_scene, _sample_textures)

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_wuhe_s02"
blocks = load_blocks(ROOT)
prof = fit_axis(blocks); profile = AxisProfile(prof)
scene, T, UV, TEX, REG = build_raycast_scene(blocks)

# 复刻 render_unwrap 单块区域: 输出图像 y 2600..2660, x 400..700 (2cm/px, z_bottom=-62.5, z_top=70)
px_m = 0.02; theta0 = 0.0
z_bottom, z_top = -62.5, 70.0
s0 = float(profile.slant(np.array([z_bottom]))[0]); s1 = float(profile.slant(np.array([z_top]))[0])
H = int(round((s1 - s0) / px_m))
_, _, rr_rng = profile.eval(np.linspace(z_bottom, z_top, 64)); r_max = float(rr_rng.max())
W = int(round(2 * np.pi * r_max / px_m))

rows = np.arange(2600, 2660); cols = np.arange(400, 700)
vv = s1 - (rows + 0.5) * px_m
zz = profile.z_from_slant(vv); cx, cy, rr = profile.eval(zz)
uu = (cols + 0.5 - W / 2) * px_m
U, ZZ = np.meshgrid(uu, zz); _, CX = np.meshgrid(uu, cx); _, CY = np.meshgrid(uu, cy); _, RR = np.meshgrid(uu, rr)
TH = theta0 + U / RR
cosT, sinT = np.cos(TH), np.sin(TH)
out_r = 1.5
rays_o = np.stack([CX + cosT*(RR+out_r), CY + sinT*(RR+out_r), ZZ, -cosT, -sinT, np.zeros_like(TH)], -1).astype(np.float32)
ans = scene.cast_rays(o3d.core.Tensor(rays_o.reshape(-1, 6)))
t_o = ans["t_hit"].numpy(); prim = ans["primitive_ids"].numpy(); bary = ans["primitive_uvs"].numpy()
INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID
hit = (prim != INVALID) & (t_o > out_r - 1.2) & (t_o < out_r + 1.0)
prim_c = np.clip(prim, 0, len(T)-1)
uv0, uv1, uv2 = UV[prim_c*3], UV[prim_c*3+1], UV[prim_c*3+2]
w1, w2 = bary[:,0:1], bary[:,1:2]; w0 = 1-w1-w2
uv_hit = uv0*w0 + uv1*w1 + uv2*w2
tex_hit = TEX[prim_c]

rgb = np.zeros((len(prim),3), np.float32)
_sample_textures([None if i is None else i for i in REG], tex_hit, uv_hit, rgb)
black = hit & (rgb.sum(1) < 60)
print(f"命中 {hit.mean()*100:.1f}%  命中但黑 {black.sum()}/{hit.sum()}")
# 打印前10个黑像素的 uv 与纹理id、图集该像素颜色
idx = np.where(black)[0][:10]
for i in idx:
    gid = tex_hit[i]; img = REG[gid]
    u, v = uv_hit[i] % 1.0
    h, w = img.shape[:2]
    x, y = int(u*(w-1)), int(v*(h-1))
    patch = img[max(0,y-3):y+4, max(0,x-3):x+4]
    print(f"tex{gid} uv=({u:.4f},{v:.4f}) xy=({x},{y}) 中心色={img[y,x]} 邻域均值={patch.mean():.0f}")
# 对照：非黑命中像素
idx2 = np.where(hit & ~black)[0][:5]
print("--- 正常像素对照 ---")
for i in idx2:
    gid = tex_hit[i]; img = REG[gid]
    u, v = uv_hit[i] % 1.0
    h, w = img.shape[:2]
    x, y = int(u*(w-1)), int(v*(h-1))
    print(f"tex{gid} uv=({u:.4f},{v:.4f}) xy=({x},{y}) 中心色={img[y,x]}")
