# -*- coding: utf-8 -*-
"""探查五河S02模型规模 + 拟合塔筒轴线与半径剖面"""
import sys, glob, os
import numpy as np
import open3d as o3d

ROOT = r"D:\KimiData\kimi\workspace\tower-unwrap\data_wuhe_s02"

def fit_circle(x, y):
    """代数圆拟合 (Kasa)"""
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx**2 + cy**2, 1e-9))
    return cx, cy, r

objs = sorted(glob.glob(os.path.join(ROOT, "*", "*.obj")))
print(f"共 {len(objs)} 个OBJ分块")

all_v = []
for p in objs:
    mesh = o3d.io.read_triangle_mesh(p)
    v = np.asarray(mesh.vertices)
    all_v.append(v)
    ntex = len(mesh.textures)
    print(f"{os.path.basename(p):28s} 顶点{len(v):>9,} 三角面{len(mesh.triangles):>9,} 纹理{ntex} UV={'有' if mesh.has_triangle_uvs() else '无'}")

V = np.vstack(all_v)
print(f"\n总顶点 {len(V):,}  总面片 {sum(len(o3d.io.read_triangle_mesh(p).triangles) for p in objs):,}" if False else f"\n总顶点 {len(V):,}")
print(f"X范围 [{V[:,0].min():.2f}, {V[:,0].max():.2f}]")
print(f"Y范围 [{V[:,1].min():.2f}, {V[:,1].max():.2f}]")
print(f"Z范围 [{V[:,2].min():.2f}, {V[:,2].max():.2f}]")

# 按高度分层拟合圆
z = V[:, 2]
zmin, zmax = np.percentile(z, [0.5, 99.5])
print(f"\n=== 分层圆拟合（每4米一层） ===")
print(f"{'z层':>8s} {'点数':>9s} {'cx':>8s} {'cy':>8s} {'r':>7s}")
rows = []
for z0 in np.arange(np.floor(zmin), zmax, 4.0):
    m = (z >= z0) & (z < z0 + 4.0)
    pts = V[m]
    if len(pts) < 200:
        continue
    # 粗中心：该层xy中位数，剔离群后迭代两次精化
    cx0, cy0 = np.median(pts[:,0]), np.median(pts[:,1])
    for _ in range(2):
        r0 = np.hypot(pts[:,0]-cx0, pts[:,1]-cy0)
        rmed = np.median(r0)
        keep = np.abs(r0 - rmed) < 1.0   # 塔面附近±1m的点
        if keep.sum() < 100:
            break
        cx0, cy0, rmed = fit_circle(pts[keep,0], pts[keep,1])
    r_all = np.hypot(pts[:,0]-cx0, pts[:,1]-cy0)
    keep = np.abs(r_all - rmed) < 1.0
    rows.append((z0+2, keep.sum(), cx0, cy0, rmed))
    print(f"{z0+2:8.1f} {keep.sum():9,} {cx0:8.3f} {cy0:8.3f} {rmed:7.3f}")

rows = np.array(rows)
if len(rows):
    cx, cy = np.median(rows[:,2]), np.median(rows[:,3])
    print(f"\n塔轴中心估计: ({cx:.3f}, {cy:.3f})  中心漂移范围 x±{rows[:,2].ptp()/2:.3f} y±{rows[:,3].ptp()/2:.3f}")
    print(f"半径范围: {rows[:,4].min():.3f} ~ {rows[:,4].max():.3f} m")
np.save(r"D:\KimiData\kimi\workspace\tower-unwrap\output\axis_profile.npy", rows)
