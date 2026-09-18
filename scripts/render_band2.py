# -*- coding: utf-8 -*-
"""参数化展开带渲染：python render_band2.py <model_root> <profile.npy> <z0> <z1> <px_m> <ss> <out.png>"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def main():
    from core.unwrap_core import load_blocks, AxisProfile, build_raycast_scene, render_unwrap
    root, prof_file = sys.argv[1], sys.argv[2]
    z0, z1, px_m, ss = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    out = sys.argv[7]
    t0 = time.time()
    blocks = load_blocks(root)
    print(f"[{time.time()-t0:.0f}s] 模型加载完成", flush=True)
    profile = AxisProfile(np.load(prof_file))
    scene_data = build_raycast_scene(blocks)
    print(f"[{time.time()-t0:.0f}s] 场景构建完成", flush=True)
    t0 = time.time()
    img, meta = render_unwrap(blocks, scene_data, profile,
                              z_bottom=z0, z_top=z1, px_m=px_m, theta0=0.0,
                              row_tile=1024, ss=ss,
                              progress=lambda i, n: print(f"tile {i}/{n} {time.time()-t0:.0f}s", flush=True))
    img.save(out)
    print(f"尺寸 {meta['W']}x{meta['H']} 黑沟判废 {meta['gutter_holes']/1e6:.2f}Mpx"
          f"（占高分 {meta['gutter_holes']/(meta['W']*meta['H']*ss*ss)*100:.1f}%）", flush=True)
    print(f"已保存 {out} 渲染 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
