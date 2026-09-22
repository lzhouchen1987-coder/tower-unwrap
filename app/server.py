# -*- coding: utf-8 -*-
"""风电混塔表面展开工具 - FastAPI 后端
流程: 加载模型(terra_obj目录) -> 塔轴拟合 -> 低清预览 -> 预览图上点选门洞 -> 设参数 -> 高清渲染导出
"""
import os, sys, time, threading, traceback, json
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.runtime_paths import app_root, data_dir, is_frozen
OUT_ROOT = os.path.join(data_dir(), "output")
STATIC = os.path.join(app_root(), "app", "static")

app = FastAPI(title="风电混塔表面展开")

# ---------------- 全局状态 ----------------
S = {
    "root": None,          # terra_obj 目录
    "blocks": None,
    "profile": None,       # AxisProfile
    "prof": None,          # (N,4) ndarray
    "scene": None,         # scene_data
    "preview": None,       # 预览元信息 {W,H,px_m,s0,s1,file}
}
JOBS = {}                  # job_id -> dict
LOCK = threading.Lock()


def _new_job(kind):
    jid = f"{kind}_{int(time.time()*1000)}"
    with LOCK:
        # 只增不删会无限膨胀：保留最近 50 条，清理更早的已完成任务
        if len(JOBS) > 50:
            old = sorted((k for k, j in JOBS.items() if j["done"]),
                         key=lambda k: JOBS[k]["t0"])
            for k in old[: len(JOBS) - 50]:
                JOBS.pop(k, None)
        JOBS[jid] = {"kind": kind, "pct": 0.0, "stage": "排队中", "done": False,
                     "result": None, "error": None, "t0": time.time()}
    return jid


def _busy():
    """有重任务（加载/预览/渲染）在进行时返回其 kind，否则 None。
    全局模型状态 S 只有一份，重任务必须互斥，防止渲染中途换模型混图。"""
    with LOCK:
        for j in JOBS.values():
            if not j["done"] and j["kind"] in ("load", "preview", "render"):
                return j["kind"]
    return None


_BUSY_MSG = {"load": "正在加载模型", "preview": "正在生成预览", "render": "正在渲染"}


import re as _re

def sanitize_name(name, default="tower"):
    """导出文件名前缀消毒：只保留中英文、数字、下划线、连字符。
    防止 ../ 等路径穿越写穿输出目录，也防 DOM 注入。"""
    s = _re.sub(r"[^\w-]+", "_", (name or "").strip())
    s = s.strip("_.-")[:40]
    return s or default


def safe_out_path(out_dir, name):
    """拼接输出路径并校验仍在 out_dir 之内（防路径穿越）"""
    base = os.path.abspath(out_dir)
    fp = os.path.abspath(os.path.join(base, name))
    if os.path.commonpath([base, fp]) != base:
        raise ValueError(f"非法文件名: {name!r}")
    return fp


def _set_job(jid, pct=None, stage=None, done=None, result=None, error=None):
    with LOCK:
        j = JOBS[jid]
        if pct is not None:
            j["pct"] = pct
        if stage is not None:
            j["stage"] = stage
        if done is not None:
            j["done"] = done
        if result is not None:
            j["result"] = result
        if error is not None:
            j["error"] = error


def _run_async(jid, fn):
    def wrap():
        try:
            fn(jid)
        except Exception as e:
            traceback.print_exc()
            _set_job(jid, done=True, error=f"{type(e).__name__}: {e}")
    threading.Thread(target=wrap, daemon=True).start()


def find_terra_obj(path):
    """接受 terra_obj 目录本身或其上级目录"""
    path = os.path.normpath(path.strip().strip('"'))
    cands = [path,
             os.path.join(path, "terra_obj"),
             os.path.join(path, "models", "pc", "0", "terra_obj"),
             os.path.join(path, "models", "pc", "terra_obj")]
    for c in cands:
        if os.path.isdir(c):
            import glob
            if glob.glob(os.path.join(c, "*", "*.obj")):
                return c
    raise FileNotFoundError(f"未找到大疆 terra_obj 分块目录: {path}")


def suggest_range(prof):
    """推荐高度范围：塔轴剖面中最长的连续层段（z间距<=6m），通常即混凝土主塔段"""
    if len(prof) == 0:
        return None
    best = [float(prof[0][0]), float(prof[0][0])]
    cur = list(best)
    for i in range(1, len(prof)):
        if prof[i][0] - prof[i - 1][0] <= 6.0:
            cur[1] = float(prof[i][0])
        else:
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
            cur = [float(prof[i][0]), float(prof[i][0])]
    if cur[1] - cur[0] > best[1] - best[0]:
        best = cur
    return [best[0] + 2.0, best[1] + 2.0]


# ---------------- API ----------------

class LoadReq(BaseModel):
    path: str


@app.post("/api/load")
def api_load(req: LoadReq):
    busy = _busy()
    if busy:
        return JSONResponse({"error": f"{_BUSY_MSG[busy]}中，请等待完成后再加载新模型"},
                            status_code=409)
    try:
        root = find_terra_obj(req.path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    jid = _new_job("load")

    def work(jid):
        from core.unwrap_core import load_blocks, fit_axis, AxisProfile, build_raycast_scene
        _set_job(jid, 0.02, "加载模型分块（含纹理匀色缓存）...")
        blocks = load_blocks(root)
        _set_job(jid, 0.55, "拟合塔轴...")
        prof = fit_axis(blocks)
        profile = AxisProfile(prof)
        _set_job(jid, 0.7, "构建光线投射场景...")
        scene = build_raycast_scene(blocks)
        S.update(root=root, blocks=blocks, profile=profile, prof=prof, scene=scene, preview=None)
        _set_job(jid, 1.0, "完成", True,
                 {"z_min": profile.z_min, "z_max": profile.z_max,
                  "r_min": float(profile.r.min()), "r_max": float(profile.r.max()),
                  "n_blocks": len(blocks), "z_suggest": suggest_range(prof)})
    _run_async(jid, work)
    return {"job_id": jid}


@app.get("/api/job/{jid}")
def api_job(jid: str):
    j = JOBS.get(jid)
    if not j:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return {k: j[k] for k in ("kind", "pct", "stage", "done", "result", "error")}


@app.get("/api/blender_status")
def api_blender_status():
    from core.blender_bake import find_blender
    p = find_blender()
    return {"found": bool(p), "path": p or ""}


@app.get("/api/profile")
def api_profile():
    if S["profile"] is None:
        return JSONResponse({"error": "尚未加载模型"}, status_code=400)
    p = S["profile"]
    return {"z_min": p.z_min, "z_max": p.z_max,
            "r_min": float(p.r.min()), "r_max": float(p.r.max())}


class PreviewReq(BaseModel):
    px_m: float = 0.02


@app.post("/api/preview")
def api_preview(req: PreviewReq):
    if S["scene"] is None:
        return JSONResponse({"error": "尚未加载模型"}, status_code=400)
    busy = _busy()
    if busy:
        return JSONResponse({"error": f"{_BUSY_MSG[busy]}中，请等待完成"}, status_code=409)
    jid = _new_job("preview")

    def work(jid):
        from core.unwrap_core import render_unwrap
        from scripts.post_flatten import push_pull_fill
        profile = S["profile"]
        out_dir = os.path.join(OUT_ROOT, "app")
        os.makedirs(out_dir, exist_ok=True)
        img, meta = render_unwrap(S["blocks"], S["scene"], profile,
                                  z_bottom=profile.z_min, z_top=profile.z_max,
                                  px_m=req.px_m, theta0=0.0, row_tile=1024, ss=1,
                                  progress=lambda i, n: _set_job(jid, 0.05 + 0.85 * i / n,
                                                                 f"渲染预览 {i}/{n}"))
        _set_job(jid, 0.92, "补洞...")
        arr = np.asarray(img)
        rgb, a = push_pull_fill(arr[..., :3], arr[..., 3])
        out = np.dstack([rgb, a])
        fp = os.path.join(out_dir, "preview.png")
        Image.fromarray(out, "RGBA").save(fp)
        S["preview"] = {"W": meta["W"], "H": meta["H"], "px_m": req.px_m,
                        "z_bottom": meta["z_bottom"], "z_top": meta["z_top"], "file": fp}
        _set_job(jid, 1.0, "完成", True,
                 {"url": "/api/file/preview.png", "W": meta["W"], "H": meta["H"]})
    _run_async(jid, work)
    return {"job_id": jid}


class RenderReq(BaseModel):
    z_bottom: float
    z_top: float
    px_m: float = 0.002
    ss: int = 2
    door_x: float = None          # 预览图像素x（None=不居中门洞）
    door_y: float = None
    split_m: float = 0.0          # >0 时每张高度（米）
    name: str = "unwrap"
    strength: float = 0.85        # 匀色强度 0~1
    clarity: float = 0.7          # 质感/清晰度增强 0~1
    engine: str = "builtin"       # builtin=内置快速 / blender=Blender烘焙（更干净，较慢）


def _log_error(where):
    """把未捕获异常的堆栈写入 data/app_error.log，便于远程排查"""
    try:
        fp = os.path.join(data_dir(), "app_error.log")
        with open(fp, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {where} =====\n")
            f.write(traceback.format_exc())
    except Exception:
        pass


@app.post("/api/render")
def api_render(req: RenderReq):
    try:
        return _api_render_impl(req)
    except Exception as e:
        _log_error("api_render")
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


def _api_render_impl(req: RenderReq):
    if S["scene"] is None:
        return JSONResponse({"error": "尚未加载模型"}, status_code=400)
    busy = _busy()
    if busy:
        return JSONResponse({"error": f"{_BUSY_MSG[busy]}中，请等待完成"}, status_code=409)
    profile = S["profile"]
    z0 = max(min(req.z_bottom, req.z_top), profile.z_min)
    z1 = min(max(req.z_bottom, req.z_top), profile.z_max)
    if z1 - z0 < 0.5:
        return JSONResponse({"error": "高度范围太小"}, status_code=400)
    # 参数上限：防极小分辨率+高超采样导致内存爆掉
    px_m = min(max(req.px_m, 0.0005), 0.05)
    ss = min(max(int(req.ss), 1), 4)
    name = sanitize_name(req.name)

    # 门洞方位角：预览图像素 -> theta
    theta0 = 0.0
    if req.door_x is not None and S["preview"]:
        pv = S["preview"]
        pv_px, W = pv["px_m"], pv["W"]
        yy = req.door_y if req.door_y is not None else pv["H"] - 1
        s1 = float(profile.slant(np.array([pv["z_top"]]))[0])
        vv = s1 - (yy + 0.5) * pv_px
        zz = float(profile.z_from_slant(np.array([vv]))[0])
        _, _, rr = profile.eval(np.array([zz]))
        rr = max(float(rr[0]), 0.05)
        uu = (req.door_x + 0.5 - W / 2) * pv_px
        theta0 = uu / rr
        if not np.isfinite(theta0):        # 剖面异常时兜底，避免 NaN 导致响应序列化 500
            theta0 = 0.0

    # 分张
    bands = []
    if req.split_m and req.split_m > 0.5:
        z = z0
        while z < z1 - 0.1:
            bands.append((z, min(z + req.split_m, z1)))
            z += req.split_m
    else:
        bands = [(z0, z1)]

    # 单张像素上限（默认 2.5 亿），超出给出明确建议而不是默默 OOM
    _, _, rr_all = profile.eval(np.linspace(z0, z1, 64))
    w_px = 2 * np.pi * float(rr_all.max()) / px_m
    for bz0, bz1 in bands:
        h_px = float(profile.slant(np.array([bz1]))[0] - profile.slant(np.array([bz0]))[0]) / px_m
        if w_px * h_px > 2.5e8:
            return JSONResponse(
                {"error": f"单张约 {w_px*h_px/1e8:.1f} 亿像素，超出上限。"
                          f"请调大像素尺寸（如 3~5mm/像素）或减小分张高度"},
                status_code=400)

    jid = _new_job("render")

    # 场景快照：渲染期间即使 S 被替换（已互斥阻止，双保险），本任务仍用当前这套模型
    snap_blocks, snap_scene, snap_profile = S["blocks"], S["scene"], profile

    def work(jid):
        from core.unwrap_core import render_unwrap
        from scripts.post_flatten import process_pipeline
        use_blender = (req.engine == "blender")
        if use_blender:
            from core.blender_bake import bake_band, find_blender
            if not find_blender():
                _set_job(jid, 1.0, "未找到 Blender", True,
                         error="未找到 Blender，请安装 Blender 4.x 后重试，或改用内置快速渲染")
                return
        out_dir = os.path.join(OUT_ROOT, "app")
        os.makedirs(out_dir, exist_ok=True)
        files = []
        n_b = len(bands)
        for bi, (bz0, bz1) in enumerate(bands):
            base = 0.02 + 0.9 * bi / n_b
            span = 0.9 / n_b
            if use_blender:
                tag0 = f"{name}_z{bz0:.0f}-{bz1:.0f}m_bake_raw.png"
                raw_png = safe_out_path(out_dir, tag0)
                arr, meta = bake_band(
                    snap_blocks, snap_profile, bz0, bz1, px_m, theta0, raw_png,
                    log_cb=lambda ln, b=base, sp=span, k=bi: _set_job(
                        jid, b + sp * 0.6, f"第{k+1}/{n_b}张 烘焙 {ln[-60:]}"))
                rgb, a = arr[..., :3], arr[..., 3]
            else:
                img, meta = render_unwrap(
                    snap_blocks, snap_scene, snap_profile,
                    z_bottom=bz0, z_top=bz1, px_m=px_m, theta0=theta0,
                    row_tile=512, ss=ss,
                    progress=lambda i, n, b=base, sp=span, k=bi: _set_job(
                        jid, b + sp * 0.75 * i / n, f"第{k+1}/{n_b}张 渲染 {i}/{n}"))
                arr = np.asarray(img)
                del img
                rgb, a = arr[..., :3], arr[..., 3]
            flat, a2 = process_pipeline(
                rgb, a, px_m,
                strength=max(0.0, min(1.0, req.strength)),
                clarity=max(0.0, min(1.0, req.clarity)),
                on_progress=lambda i, n, b=base, sp=span, k=bi: _set_job(
                    jid, b + sp * (0.78 + 0.18 * i / n), f"第{k+1}/{n_b}张 匀色 {i}/{n}"))
            del rgb, arr
            if use_blender:
                try:
                    os.remove(raw_png)      # 烘焙中间图很大，后处理后删除
                except OSError:
                    pass
            out = np.dstack([flat, a2])
            tag = f"{name}_z{bz0:.0f}-{bz1:.0f}m_{px_m*1000:.0f}mm.png"
            fp = safe_out_path(out_dir, tag)
            _set_job(jid, base + span * 0.97, f"第{bi+1}/{n_b}张 保存...")
            Image.fromarray(out, "RGBA").save(fp)
            del out, flat
            files.append({"name": tag, "url": f"/api/file/{tag}",
                          "W": meta["W"], "H": meta["H"]})
        _set_job(jid, 1.0, "完成", True,
                 {"files": files, "theta0": theta0, "z": [z0, z1]})
    _run_async(jid, work)
    return {"job_id": jid, "theta0": theta0, "bands": len(bands), "name": name}


@app.get("/api/file/{name}")
def api_file(name: str, dl: int = 0):
    try:
        fp = safe_out_path(os.path.join(OUT_ROOT, "app"), os.path.basename(name))
    except ValueError:
        return JSONResponse({"error": "非法文件名"}, status_code=400)
    if not os.path.exists(fp):
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    if dl:                       # 浏览器模式下强制作为附件下载
        return FileResponse(fp, filename=os.path.basename(name))
    return FileResponse(fp)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
