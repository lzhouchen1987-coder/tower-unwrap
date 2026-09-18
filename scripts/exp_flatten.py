# -*- coding: utf-8 -*-
"""输出端匀色参数实验：多组 flatten_lab 参数，生成整带视图+原尺寸裁切对比"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import cv2
from PIL import Image
from scripts.post_flatten import fill_small_holes, flatten_lab
Image.MAX_IMAGE_PIXELS = None

SRC = r"output\yiyang08_band_2mm.png"
PX_M = 0.002
# 裁切区域（与蓝色补丁样本点对应）: (cx, cy) 中心, 1600x1200
CROPS = [(6052, 4009), (2895, 2434)]

PARAM_SETS = [
    ("s030_l100_c100", dict(sigma_m=0.30, l_strength=1.0, c_strength=1.0)),
    ("s050_l100_c100", dict(sigma_m=0.50, l_strength=1.0, c_strength=1.0)),
]


def main():
    arr = np.asarray(Image.open(SRC))
    rgb, alpha = arr[..., :3], arr[..., 3]
    print("填补小孔洞...", flush=True)
    rgb, alpha = fill_small_holes(rgb, alpha)
    # 原图裁切留档
    for i, (cx, cy) in enumerate(CROPS):
        Image.fromarray(rgb[max(cy-600,0):cy+600, max(cx-800,0):cx+800]).save(
            rf"output\exp_raw_crop{i}.jpg", quality=90)
    for tag, kw in PARAM_SETS:
        print(f"--- {tag} {kw}", flush=True)
        flat = flatten_lab(rgb, alpha, px_m=PX_M, **kw)
        Image.fromarray(flat).resize((1400, int(flat.shape[0]*1400/flat.shape[1]))).save(
            rf"output\exp_{tag}_view.jpg", quality=88)
        for i, (cx, cy) in enumerate(CROPS):
            Image.fromarray(flat[max(cy-600,0):cy+600, max(cx-800,0):cx+800]).save(
                rf"output\exp_{tag}_crop{i}.jpg", quality=90)
        d = flat[..., 2].astype(np.int16) - flat[..., 0].astype(np.int16)
        dv = d[alpha >= 128]
        lum = flat.sum(-1) / 3.0
        lv = lum[alpha >= 128]
        print(f"   B-R 中位={np.median(dv):.0f} 90分位={np.percentile(dv,90):.0f} | "
              f"亮度 10~90分位={np.percentile(lv,10):.0f}~{np.percentile(lv,90):.0f}", flush=True)
        del flat


if __name__ == "__main__":
    main()
