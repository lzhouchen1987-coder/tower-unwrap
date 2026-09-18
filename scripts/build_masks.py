# -*- coding: utf-8 -*-
"""为模型所有图集构建"有效照片内容"掩膜（剔除黑沟与楔状毛边），供渲染时判废采样点
掩膜与 v6 纹理缓存同名：tex_cache/{key}_mask.png，1024x1024，255=有效"""
import sys, os, time, hashlib, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
from PIL import Image
from core.unwrap_core import _parse_mtl_textures
import glob

Image.MAX_IMAGE_PIXELS = None
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "tex_cache")
MASK_RES = 1024


def cache_key(img_file):
    st = os.stat(img_file)
    return hashlib.md5(f"v6|{img_file}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def build_mask(img_file, out_path):
    """1/8分辨率判定有效区→轻腐蚀去楔边→存1024²掩膜"""
    with Image.open(img_file) as im:
        full = im.size
    ds = max(1, round(max(full) / MASK_RES))
    with Image.open(img_file) as im:
        im = im.resize((full[0] // ds, full[1] // ds), Image.BILINEAR)
        small = np.asarray(im.convert("RGB"))
    valid = (small.sum(-1) > 105).astype(np.uint8)
    # 轻腐蚀去楔状毛边（3x3 ≈ 8192²上约16px）
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    # 去掉小碎岛（面积<0.02%的孤立点，通常是噪点）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(valid, 8)
    keep = np.zeros_like(valid)
    min_area = max(16, int(valid.shape[0] * valid.shape[1] * 0.0002))
    for i in range(1, n):
        if stats[i, 4] >= min_area:
            keep[lab == i] = 1
    keep = cv2.resize(keep, (MASK_RES, MASK_RES), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(keep * 255).save(out_path)


def main(root):
    objs = sorted(glob.glob(os.path.join(root, "*", "*.obj")))
    files = set()
    for p in objs:
        texmap = _parse_mtl_textures(p, fix_gutters=False)
        for idx, e in texmap.items():
            if e:
                files.add(e[0])
    files = sorted(files)
    print(f"共 {len(files)} 张图集", flush=True)
    t0 = time.time()
    done = skip = 0
    for f in files:
        out = os.path.join(CACHE_DIR, cache_key(f) + "_mask.png")
        if os.path.exists(out):
            skip += 1
            continue
        try:
            build_mask(f, out)
            done += 1
            if done % 10 == 0:
                print(f"  已建 {done} 跳过 {skip} 用时 {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  [失败] {os.path.basename(f)}: {e}", flush=True)
    print(f"完成：新建 {done}，已存在 {skip}，总用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"D:\KimiData\kimi\workspace\tower-unwrap\data_yiyang_08")
