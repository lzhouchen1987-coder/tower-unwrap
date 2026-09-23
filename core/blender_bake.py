# -*- coding: utf-8 -*-
"""Blender 烘焙渲染引擎：把高模贴图烘焙到干净的低模圆锥筒上（低模→展UV→烘焙）。
自动查找本机 Blender，生成 job.json，无头调用 bake_unwrap.py，读回展开图原图。"""
import glob
import json
import os
import subprocess
import tempfile

import numpy as np

_BAKE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "blender", "bake_unwrap.py")


def find_blender():
    """查找 blender.exe：环境变量 TOWER_BLENDER > 常见安装目录 > PATH"""
    env = os.environ.get("TOWER_BLENDER")
    if env and os.path.exists(env):
        return env
    cands = []
    for root in (r"C:\Program Files\Blender Foundation",
                 r"D:\Program Files\Blender Foundation",
                 r"E:\Program Files\Blender Foundation"):
        cands += sorted(glob.glob(os.path.join(root, "*", "blender.exe")), reverse=True)
    for c in cands:
        if os.path.exists(c):
            return c
    for p in os.environ.get("PATH", "").split(os.pathsep):
        c = os.path.join(p, "blender.exe")
        if os.path.exists(c):
            return c
    return None


def _band_meta(profile, z0, z1, px_m):
    """与内置渲染完全一致的画布尺寸计算"""
    s0 = float(profile.slant(np.array([z0]))[0])
    s1 = float(profile.slant(np.array([z1]))[0])
    H = int(round((s1 - s0) / px_m))
    _, _, rr = profile.eval(np.linspace(z0, z1, 64))
    r_max = float(rr.max())
    W = int(round(2 * np.pi * r_max / px_m))
    return W, H, s0, s1, r_max


def bake_band(blocks, profile, z0, z1, px_m, theta0, out_png,
              blender_path=None, log_cb=None, timeout_s=7200, scale=2):
    """烘焙一个高度带，返回 (RGBA uint8 数组, meta)。
    烘焙图已按展开图布局（u=方位角@门洞居中，v=斜高），按 scale 超采样烘焙后
    归一化降采样回目标尺寸，再把锥形顶部超出本地周长的角落置为透明。"""
    bp = blender_path or find_blender()
    if not bp:
        raise RuntimeError("未找到 Blender，请安装 Blender 4.x 或设置环境变量 TOWER_BLENDER")
    script = _BAKE_SCRIPT
    if getattr(__import__("sys"), "frozen", False):
        script = os.path.join(__import__("sys")._MEIPASS, "blender", "bake_unwrap.py")
    W, H, s0, s1, r_max = _band_meta(profile, z0, z1, px_m)
    # 超采样上限：烘焙目标为float显存/内存，控制在 ~2.5亿像素内
    while scale > 1 and W * H * scale * scale > 2.5e8:
        scale -= 1

    prof_rows = [[float(z), float(cx), float(cy), float(r)]
                 for z, cx, cy, r in zip(profile.z, profile.cx, profile.cy, profile.r)]
    out_png = os.path.abspath(out_png)
    job = {
        "objs": [b["path"] for b in blocks],
        "profile": prof_rows,
        "z0": float(z0), "z1": float(z1),
        "px_m": float(px_m), "theta0": float(theta0),
        "W": W, "H": H, "scale": scale, "out_png": out_png,
    }
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(job, f)
        job_file = f.name
    try:
        cmd = [bp, "--background", "--factory-startup",
               "--python", script, "--", job_file]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        log_lines = []
        done_seen = False
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                log_lines.append(line)
                if "[bake] DONE" in line:
                    done_seen = True
                if log_cb:
                    log_cb(line)
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("Blender 烘焙超时")
        # 完整烘焙日志落盘，便于排查
        try:
            from core.runtime_paths import data_dir
            with open(os.path.join(data_dir(), "bake_last.log"), "w",
                      encoding="utf-8") as lf:
                lf.write("\n".join(log_lines))
        except Exception:
            pass
        out_ok = os.path.exists(job["out_png"]) and os.path.getsize(job["out_png"]) > 0
        if proc.returncode != 0:
            if done_seen and out_ok:
                # 输出已完整写盘后 Blender 在退出清理阶段崩溃：视为成功
                if log_cb:
                    log_cb(f"[bake] Blender 退出码 {proc.returncode}，但结果已完整输出，继续")
            else:
                tail = " | ".join(log_lines[-8:])
                raise RuntimeError(
                    f"Blender 烘焙失败（退出码 {proc.returncode}）。日志末尾：{tail}")
        if not out_ok:
            raise RuntimeError("Blender 未输出烘焙结果")
    finally:
        try:
            os.unlink(job_file)
        except OSError:
            pass

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    arr = np.asarray(Image.open(job["out_png"]).convert("RGBA")).copy()
    # 超采样降采样：归一化盒式（空洞像素不参与平均），W2/H2 恢复目标尺寸
    Hb, Wb = arr.shape[:2]
    sc = Wb // W
    if sc > 1 and Hb // sc == H:
        import cv2
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3:4].astype(np.float32) / 255.0
        num = cv2.resize(rgb * a, (W, H), interpolation=cv2.INTER_AREA)
        den = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
        arr = np.dstack([np.clip(num / np.maximum(den[..., None], 1e-3), 0, 255),
                         np.clip(den * 255, 0, 255)]).astype(np.uint8)
    # 锥形对齐：每行有效半宽 = pi*r(z)/px_m，超出部分置透明（与内置渲染一致）
    H2, W2 = arr.shape[:2]
    rows = np.arange(H2)
    vv = s1 - (rows + 0.5) * (s1 - s0) / H2          # 顶行=最高斜高
    zz = profile.z_from_slant(vv)
    _, _, rr = profile.eval(zz)
    half = np.pi * rr / px_m                          # 有效半宽（像素）
    cols = np.abs(np.arange(W2) + 0.5 - W2 / 2)
    valid = cols[None, :] <= half[:, None]
    arr[..., 3] = np.where(valid, arr[..., 3], 0).astype(np.uint8)
    meta = dict(W=W2, H=H2, width_m=2 * np.pi * r_max, height_m=s1 - s0,
                gutter_holes=0)
    return arr, meta
