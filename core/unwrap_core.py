# -*- coding: utf-8 -*-
"""
风塔混塔表面等距展开核心模块
- 读取大疆智图 terra_obj 分块（obj+mtl+jpg）
- 拟合塔轴中心 cx(z), cy(z) 与半径剖面 r(z)
- 以 (弧长, 斜高) 等距参数化，用光线投射把照片纹理采样到展开图
"""
import os, glob
import numpy as np
import open3d as o3d
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ---------------- 模型加载 ----------------

def _equalize_atlas(img, thresh=105, erode_px=15):
    """照片补丁级匀色（逐通道）：腐蚀断开补丁→逐补丁求RGB均值→增益拉到中位数→标签回传
    返回 (匀色后图像, 本图集补丁RGB中位数[3])"""
    import cv2
    from scipy.ndimage import distance_transform_edt
    valid = (img.sum(-1) > thresh).astype(np.uint8)
    if valid.mean() < 0.01:
        return img, None
    core = cv2.erode(valid, np.ones((2 * erode_px + 1,) * 2, np.uint8))
    n, labels = cv2.connectedComponents(core, connectivity=8)
    if n < 3:
        return img, None
    sizes = np.bincount(labels.ravel(), minlength=n)
    ok = sizes > 20000
    ok[0] = False
    if ok.sum() < 2:
        return img, None
    lab = labels.ravel()
    sums = np.stack([np.bincount(lab, weights=img[..., c].ravel().astype(np.float64), minlength=n)
                     for c in range(3)], axis=1)
    means = sums / np.maximum(sizes, 1)[:, None]
    target = np.median(means[ok], axis=0)              # 逐通道中位数
    gain_lab = np.clip(target[None, :] / np.maximum(means, 1.0), 0.6, 1.7)
    gain_lab[~ok] = 1.0
    # 腐蚀区外的有效像素取最近补丁的增益
    unlabeled = (valid == 1) & (labels == 0)
    lab_full = labels
    if unlabeled.any():
        _, inds = distance_transform_edt(labels == 0, return_indices=True)
        lab_full = labels.copy()
        lab_full[unlabeled] = labels[inds[0][unlabeled], inds[1][unlabeled]]
    # 1/4分辨率构建并平滑三通道增益图，再上采样（省内存）
    lab_q = lab_full[::4, ::4]
    gain_q = gain_lab[lab_q].astype(np.float32)
    vf_q = valid[::4, ::4].astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(vf_q, (0, 0), 2), 1e-3)
    gm_q = np.stack([cv2.GaussianBlur(gain_q[..., c] * vf_q, (0, 0), 2) / den for c in range(3)], -1)
    gm = np.stack([cv2.resize(gm_q[..., c], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                   for c in range(3)], -1)
    gm = np.clip(gm, 0.6, 1.7)
    gm[valid == 0] = 1.0
    # 原地相乘复用 gm 内存，避免 (8192,8192,3) float32 峰值翻倍
    np.multiply(img, gm, out=gm, casting='unsafe')
    np.clip(gm, 0, 255, out=gm)
    return gm.astype(np.uint8), target.tolist()


def _dilate_atlas(img, thresh=105):
    """用 EDT 最近邻把图集岛屿颜色一次性填充到黑色 gutter 区域"""
    from scipy.ndimage import distance_transform_edt
    dark = img.sum(-1) <= thresh
    if dark.mean() < 0.0005:
        return img
    _, inds = distance_transform_edt(dark, return_indices=True)
    out = img.copy()
    out[dark] = img[inds[0][dark], inds[1][dark]]
    return out


def _ensure_texture_cache(img_file, cache_dir):
    """确保 匀色+gutter填充 的缓存存在，返回 (缓存路径, (h,w))，不常驻内存"""
    import hashlib
    st = os.stat(img_file)
    key = hashlib.md5(f"v6|{img_file}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, key + ".jpg")
    meta_file = os.path.join(cache_dir, key + ".json")
    med = None
    if not os.path.exists(cache):
        img = np.asarray(Image.open(img_file).convert("RGB"))
        img, med = _equalize_atlas(img)  # 先孤岛匀色
        img = _dilate_atlas(img)         # 再填充gutter
        Image.fromarray(img).save(cache, quality=92)
        del img
        import json
        with open(meta_file, "w") as f:
            json.dump({"med": med}, f)
    elif os.path.exists(meta_file):
        import json
        try:
            with open(meta_file) as f:
                med = json.load(f).get("med")
        except Exception:
            med = None
    if med is None:
        # 老缓存缺统计：从匀色后缓存图快速估计（降采样逐通道中位数）
        try:
            with Image.open(cache) as im:
                im.thumbnail((1024, 1024))
                small = np.asarray(im.convert("RGB"))
            valid = small.sum(-1) > 105
            if valid.any():
                med = np.median(small[valid].reshape(-1, 3), axis=0).tolist()
                import json
                with open(meta_file, "w") as f:
                    json.dump({"med": med}, f)
        except Exception:
            med = None
    with Image.open(cache) as im:
        h, w = im.size[1], im.size[0]
    return cache, (h, w), med


class MaskPool:
    """图集有效区掩膜池：1024² uint8(0/1)，常驻内存（每张1MB），无掩膜时视为全有效"""

    def __init__(self, registry):
        self.reg = registry
        self.masks = {}   # gid -> np.ndarray(uint8 0/1) or None

    def get(self, gid):
        if gid in self.masks:
            return self.masks[gid]
        entry = self.reg[gid]
        m = None
        if entry:
            mp = entry[0].replace(".jpg", "_mask.png")
            if os.path.exists(mp):
                m = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8)
        self.masks[gid] = m
        return m


class TexturePool:
    """纹理 LRU 池：按字节上限驻留，超出自动逐出最久未用"""

    def __init__(self, registry, max_bytes=6 << 30):
        self.reg = registry          # list of (path,(h,w)) or None
        self.max_bytes = max_bytes
        self.cache = {}              # gid -> ndarray
        self.lru = []                # 最近使用顺序
        self.bytes = 0

    def get(self, gid):
        if gid in self.cache:
            self.lru.remove(gid)
            self.lru.append(gid)
            return self.cache[gid]
        entry = self.reg[gid]
        if entry is None:
            return None
        path, (h, w) = entry[0], entry[1]
        img = np.asarray(Image.open(path).convert("RGB"))
        need = h * w * 3
        while self.bytes + need > self.max_bytes and self.lru:
            old = self.lru.pop(0)
            o = self.cache.pop(old)
            self.bytes -= o.nbytes
        self.cache[gid] = img
        self.lru.append(gid)
        self.bytes += need
        return img


def _pool_ensure(args):
    return _ensure_texture_cache(*args)


def _parse_mtl_textures(obj_path, fix_gutters=True, cache_dir=None):
    """解析 obj 同目录 mtl，按 newmtl 数字名 -> map_Kd 建立纹理索引表"""
    mtl_path = None
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib"):
                mtl_path = os.path.join(os.path.dirname(obj_path), line.split(None, 1)[1].strip())
                break
    texmap = {}
    cur = None
    if mtl_path and os.path.exists(mtl_path):
        with open(mtl_path, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("newmtl"):
                    try:
                        cur = int(s.split(None, 1)[1])
                    except ValueError:
                        cur = None
                elif s.startswith("map_Kd") and cur is not None:
                    texmap[cur] = os.path.join(os.path.dirname(mtl_path), s.split(None, 1)[1])
    if cache_dir is None:
        from core.runtime_paths import data_dir
        cache_dir = os.path.join(data_dir(), "tex_cache")
    images = {}
    if fix_gutters and len(texmap) > 1:
        from concurrent.futures import ProcessPoolExecutor
        keys = list(texmap.keys())
        args = [(texmap[k], cache_dir) for k in keys]
        with ProcessPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(_pool_ensure, args))
        for idx, res in zip(keys, results):
            images[idx] = res
        return images
    for idx, img_file in texmap.items():
        try:
            if fix_gutters:
                images[idx] = _ensure_texture_cache(img_file, cache_dir)
            else:
                with Image.open(img_file) as im:
                    images[idx] = (img_file, (im.size[1], im.size[0]), None)
        except Exception as e:
            print(f"   [警告] 纹理 {img_file} 读取失败: {e}", flush=True)
            images[idx] = None
    return images


def load_obj_fast(path):
    """自写OBJ解析（不加载贴图，内存安全）。返回 顶点/三角面/逐角UV/逐面材质"""
    verts, uvs, faces, face_uv, tri_mat = [], [], [], [], []
    cur_mat = -1
    with open(path, "rb") as f:
        for line in f:
            if line[:2] == b"v ":
                verts.append(line[2:].split())
            elif line[:3] == b"vt ":
                uvs.append(line[3:].split()[:2])
            elif line[:2] == b"f ":
                idx, tuv = [], []
                for p in line[2:].split():
                    q = p.split(b"/")
                    idx.append(int(q[0]) - 1)
                    tuv.append(int(q[1]) - 1 if len(q) > 1 and q[1] else 0)
                for k in range(1, len(idx) - 1):   # 扇形三角化（DJI均为三角面）
                    faces.append((idx[0], idx[k], idx[k + 1]))
                    face_uv.append((tuv[0], tuv[k], tuv[k + 1]))
                    tri_mat.append(cur_mat)
            elif line[:6] == b"usemtl":
                cur_mat = int(line.split(None, 1)[1])
    V = np.array(verts, dtype=np.float32)
    UVall = np.array(uvs, dtype=np.float32) if uvs else np.zeros((1, 2), np.float32)
    F = np.array(faces, dtype=np.int32)
    FUV = np.array(face_uv, dtype=np.int64)
    tri_uvs = UVall[np.clip(FUV, 0, len(UVall) - 1)].reshape(-1, 2).astype(np.float32)
    return V, F, tri_uvs, np.array(tri_mat, np.int32)


def load_blocks(root):
    """读取 terra_obj 目录下所有分块（几何自解析，纹理走 mtl+缓存）"""
    objs = sorted(glob.glob(os.path.join(os.path.abspath(root), "*", "*.obj")))
    if not objs:
        raise FileNotFoundError(f"未在 {root} 找到 obj 分块")
    blocks = []
    for p in objs:
        print(f"   加载 {os.path.basename(p)} ...", flush=True)
        V, F, tri_uvs, tri_mat = load_obj_fast(p)
        textures = _parse_mtl_textures(p)
        blocks.append({
            "path": p,
            "vertices": V,
            "triangles": F,
            "tri_uvs": tri_uvs,      # (F*3,2)
            "tri_mat": tri_mat,
            "textures": textures,
        })
    return blocks


# ---------------- 塔轴拟合 ----------------

def _fit_circle(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-9))
    return cx, cy, r


def fit_axis(blocks, z_step=4.0, band=1.0, min_pts=100):
    """分层圆拟合，返回 profile: (N,4) [z, cx, cy, r]，按z升序"""
    V = np.vstack([b["vertices"] for b in blocks])
    z = V[:, 2]
    zmin, zmax = np.percentile(z, [0.2, 99.9])
    rows = []
    for z0 in np.arange(np.floor(zmin), zmax, z_step):
        m = (z >= z0) & (z < z0 + z_step)
        pts = V[m]
        if len(pts) < min_pts * 3:
            continue
        cx0, cy0 = np.median(pts[:, 0]), np.median(pts[:, 1])
        ok = False
        for _ in range(3):
            r0 = np.hypot(pts[:, 0] - cx0, pts[:, 1] - cy0)
            rmed = np.median(r0)
            keep = np.abs(r0 - rmed) < band
            if keep.sum() < min_pts:
                break
            cx0, cy0, rmed = _fit_circle(pts[keep, 0], pts[keep, 1])
            ok = True
        if ok:
            rows.append((z0 + z_step / 2, cx0, cy0, rmed))
    prof = np.array(rows)
    # 剔除明显异常层（半径突变>1.5倍邻层，通常是地面杂物）；首末层与相邻层比较
    if len(prof) >= 5:
        r = prof[:, 3]
        med = np.median(r)
        good = np.ones(len(prof), bool)
        for i in range(len(prof)):
            if i == 0:
                neigh = r[1]
            elif i == len(prof) - 1:
                neigh = r[-2]
            else:
                neigh = 0.5 * (r[i - 1] + r[i + 1])
            if r[i] > 1.5 * neigh or r[i] > 3 * med:
                good[i] = False
        prof = prof[good]
    return prof


class AxisProfile:
    """cx(z), cy(z), r(z) 插值 + 斜高 s(z)"""

    def __init__(self, prof):
        self.z = prof[:, 0]
        self.cx = np.interp(self.z, self.z, prof[:, 1])
        self.cy = np.interp(self.z, self.z, prof[:, 2])
        self.r = prof[:, 3]
        # 斜高：沿锥面母线的累计长度
        dz = np.diff(self.z)
        dr = np.diff(self.r)
        ds = np.sqrt(dz ** 2 + dr ** 2)
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.z_min, self.z_max = float(self.z[0]), float(self.z[-1])
        self.r_max = float(self.r.max())

    def eval(self, zq):
        zq = np.asarray(zq)
        cx = np.interp(zq, self.z, self.cx)
        cy = np.interp(zq, self.z, self.cy)
        r = np.interp(zq, self.z, self.r)
        return cx, cy, r

    def slant(self, zq):
        return np.interp(zq, self.z, self.s)

    def z_from_slant(self, sq):
        return np.interp(sq, self.s, self.z)


# ---------------- 展开渲染 ----------------

def build_raycast_scene(blocks):
    """合并所有分块为 o3d.t 网格并构建光线投射场景；返回 scene 与逐三角形纹理映射表"""
    verts, tris, tri_uvs, tri_tex = [], [], [], []
    v_off = 0
    tex_registry = []  # 全局纹理表: [(block_idx, local_tex_idx, image)]
    tex_map = {}       # (block_idx, local_idx) -> global_tex_idx
    for bi, b in enumerate(blocks):
        v, t, uv, tm = b["vertices"], b["triangles"], b["tri_uvs"], b["tri_mat"]
        verts.append(v)
        tris.append(t + v_off)
        tri_uvs.append(uv)
        gids = np.zeros(len(t), np.int32)
        for li in np.unique(tm):
            key = (bi, int(li))
            if key not in tex_map:
                img = b["textures"].get(int(li))
                tex_map[key] = len(tex_registry)
                tex_registry.append(img)
            gids[tm == li] = tex_map[key]
        tri_tex.append(gids)
        v_off += len(v)
    V = np.vstack(verts)
    T = np.vstack(tris)
    UV = np.vstack(tri_uvs)          # (F*3,2)
    TEX = np.concatenate(tri_tex)    # (F,)

    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(V, o3d.core.float32)
    tm.triangle.indices = o3d.core.Tensor(T, o3d.core.int32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    return scene, T, UV, TEX, tex_registry


def _sample_textures(pool, hit_tex, uv_hit, out_rgb, gains=None, mask_pool=None, out_valid=None):
    """按全局纹理id分组，双线性采样填充 out_rgb(N,3)。pool: TexturePool；gains: 逐图集全局匀色增益；
    mask_pool 提供图集有效区掩膜，采样点落在黑沟/楔边时 out_valid=False（交由输出端补洞）"""
    for gid in np.unique(hit_tex):
        m = hit_tex == gid
        img = pool.get(int(gid))
        if img is None:
            out_rgb[m] = 200
            continue
        h, w = img.shape[:2]
        u = uv_hit[m, 0] % 1.0
        # OBJ规范 v=0 在纹理底部（open3d加载时已翻转）：采样图像行需用 1-v
        v = 1.0 - (uv_hit[m, 1] % 1.0)
        # open3d OBJ: v轴向下，与图像行一致
        x = u * (w - 1)
        y = v * (h - 1)
        x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
        x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1); y0 = np.clip(y0, 0, h - 1)
        fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
        c00 = img[y0, x0].astype(np.float32)
        c10 = img[y0, x1].astype(np.float32)
        c01 = img[y1, x0].astype(np.float32)
        c11 = img[y1, x1].astype(np.float32)
        out_rgb[m] = (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
                      + c01 * (1 - fx) * fy + c11 * fx * fy)
        if gains is not None:
            out_rgb[m] *= gains[int(gid)][None, :]
        if mask_pool is not None and out_valid is not None:
            vmask = mask_pool.get(int(gid))
            if vmask is not None:
                mh, mw = vmask.shape
                mx = np.clip((u * mw).astype(np.int64), 0, mw - 1)
                my = np.clip((v * mh).astype(np.int64), 0, mh - 1)
                idx = np.where(m)[0]
                out_valid[idx[vmask[my, mx] == 0]] = False


def render_unwrap(blocks, scene_data, profile: AxisProfile,
                  z_bottom, z_top, px_m=0.01, theta0=0.0,
                  row_tile=2048, ss=1, progress=None):
    """
    渲染展开图。
    - u = r(z)*(theta-theta0)：横向弧长（米），图像横向中心为 u=0（门洞居中时 theta0=门方位角）
    - v = slant(z)：纵向斜高（米），自下而上
    - 画布宽 = 2*pi*r_max/px_m（底部周长），顶部行居中留空
    - ss: 超采样倍数，内部按 px_m/ss 渲染再盒式降采样，抗锯齿消除碎块感
    返回 RGBA uint8 图像（PIL）与元信息 dict
    """
    import cv2
    scene, T, UV, TEX, tex_registry = scene_data
    pool = TexturePool(tex_registry)
    mask_pool = MaskPool(tex_registry)
    # 全局匀色：跨图集把各自中位色拉到全模型中位数（逐通道）
    meds = np.full((len(tex_registry), 3), np.nan)
    for i, e in enumerate(tex_registry):
        if e and e[2]:
            meds[i] = np.asarray(e[2], float)[:3]
    gains = None
    if np.isfinite(meds[:, 0]).sum() >= 2:
        global_med = np.nanmedian(meds, axis=0)
        safe = np.where(np.isfinite(meds), meds, global_med[None, :])
        gains = np.clip(global_med[None, :] / safe, 0.5, 2.0)
        gains[~np.isfinite(meds)] = 1.0

    s0 = float(profile.slant(np.array([z_bottom]))[0])
    s1 = float(profile.slant(np.array([z_top]))[0])
    H = int(round((s1 - s0) / px_m))
    # 画布宽度按高度范围内最大半径（避免地面层把画布撑宽）
    _, _, rr_rng = profile.eval(np.linspace(z_bottom, z_top, 64))
    r_max = float(rr_rng.max())
    W = int(round(2 * np.pi * r_max / px_m))
    if H <= 0 or W <= 0:
        raise ValueError("高度范围无效")

    px_hi = px_m / ss
    W2, H2 = W * ss, H * ss

    img = np.zeros((H, W, 4), np.uint8)
    n_tiles = (H + row_tile - 1) // row_tile
    hole_stats = [0]     # 掩膜判废像素数（高分辨率）
    INVALID = o3d.t.geometry.RaycastingScene.INVALID_ID
    out_r = 1.5      # 外侧投射起点超出拟合面的距离
    band_out = 1.0   # 命中点允许陷入拟合面内侧的深度
    band_clut = 1.2  # 命中点允许突出拟合面外侧的距离（爬梯等）

    for ti in range(n_tiles):
        r0, r1 = ti * row_tile, min(H, (ti + 1) * row_tile)
        rows = np.arange(r0 * ss, r1 * ss)      # 高分辨率行
        # 像素 -> (u 弧长, v 斜高)
        vv = s1 - (rows + 0.5) * px_hi          # 顶行是最高处
        zz = profile.z_from_slant(vv)
        cx, cy, rr = profile.eval(zz)
        uu = (np.arange(W2) + 0.5 - W2 / 2) * px_hi
        U, ZZ = np.meshgrid(uu, zz)
        _, CX = np.meshgrid(uu, cx)
        _, CY = np.meshgrid(uu, cy)
        _, RR = np.meshgrid(uu, rr)
        RR = np.maximum(RR, 0.05)
        TH = theta0 + U / RR                    # theta = theta0 + u/r(z)
        # 该行 u 超出本地周长一半的位置无效（锥顶行宽小于画布）
        valid = np.abs(U) <= np.pi * RR
        vflat = valid.reshape(-1)
        cosT, sinT = np.cos(TH), np.sin(TH)

        N = U.size
        prim_f = np.full(N, INVALID, np.uint32)
        bary_f = np.zeros((N, 2), np.float32)
        hit_f = np.zeros(N, bool)

        # —— 第一遍：外侧向轴心投（取塔面外壁，抗凹陷） ——
        rays_o = np.stack([CX + cosT * (RR + out_r), CY + sinT * (RR + out_r), ZZ,
                           -cosT, -sinT, np.zeros_like(TH)], -1).astype(np.float32)
        rays_o[~valid] = 0.0
        ans = scene.cast_rays(o3d.core.Tensor(rays_o.reshape(-1, 6)))
        t_o = ans["t_hit"].numpy()
        prim_o = ans["primitive_ids"].numpy()
        bary_o = ans["primitive_uvs"].numpy()
        # 命中点半径 = RR + out_r - t，须在 [RR-band_out, RR+band_clut] 内
        ok_o = (prim_o != INVALID) & vflat \
               & (t_o > out_r - band_clut) & (t_o < out_r + band_out)
        hit_f[ok_o] = True
        prim_f[ok_o] = prim_o[ok_o]
        bary_f[ok_o] = bary_o[ok_o]

        # —— 第二遍：外侧未命中的像素，从内侧向外补投 ——
        miss = vflat & ~hit_f
        if miss.any():
            rays_i = np.stack([CX + cosT * (RR - 0.3), CY + sinT * (RR - 0.3), ZZ,
                               cosT, sinT, np.zeros_like(TH)], -1).astype(np.float32)
            sub = rays_i.reshape(-1, 6)[miss]
            ans2 = scene.cast_rays(o3d.core.Tensor(sub))
            t_i = ans2["t_hit"].numpy()
            prim_i = ans2["primitive_ids"].numpy()
            bary_i = ans2["primitive_uvs"].numpy()
            ok_i = (prim_i != INVALID) & (t_i < 0.3 + band_clut)
            midx = np.where(miss)[0]
            sel = midx[ok_i]
            hit_f[sel] = True
            prim_f[sel] = prim_i[ok_i]
            bary_f[sel] = bary_i[ok_i]

        hit = hit_f
        prim = prim_f
        bary = bary_f
        prim_c = np.clip(prim, 0, len(T) - 1)
        uv0 = UV[prim_c * 3 + 0]; uv1 = UV[prim_c * 3 + 1]; uv2 = UV[prim_c * 3 + 2]
        w1 = bary[:, 0:1]; w2 = bary[:, 1:2]; w0 = 1.0 - w1 - w2
        uv_hit = uv0 * w0 + uv1 * w1 + uv2 * w2
        tex_hit = TEX[prim_c]

        out = np.zeros((N, 3), np.uint8)
        hit_idx = np.where(hit)[0]
        if len(hit_idx):
            sel_rgb = np.zeros((len(hit_idx), 3), np.float32)
            sel_valid = np.ones(len(hit_idx), bool)
            _sample_textures(pool, tex_hit[hit_idx], uv_hit[hit_idx], sel_rgb, gains,
                             mask_pool, sel_valid)
            good = hit_idx[sel_valid]
            out[good] = np.clip(sel_rgb[sel_valid], 0, 255).astype(np.uint8)
            hit[hit_idx[~sel_valid]] = False      # 采样落在黑沟/楔边 → 置为空洞，输出端补
            hole_stats[0] += int((~sel_valid).sum())
        hi_rgb = out.reshape(r1 * ss - r0 * ss, W2, 3)
        hi_a = hit.reshape(r1 * ss - r0 * ss, W2).astype(np.float32)
        if ss > 1:
            # 归一化降采样：空洞像素不参与平均，避免黑边
            num = cv2.resize(hi_rgb.astype(np.float32) * hi_a[..., None], (W, r1 - r0),
                             interpolation=cv2.INTER_AREA)
            den = cv2.resize(hi_a, (W, r1 - r0), interpolation=cv2.INTER_AREA)
            hi_rgb = np.clip(num / np.maximum(den[..., None], 1e-3), 0, 255).astype(np.uint8)
            hi_a = den
        tile = img[r0:r1]
        tile[..., :3] = hi_rgb
        tile[..., 3] = np.clip(hi_a * 255, 0, 255).astype(np.uint8)
        if progress:
            progress(ti + 1, n_tiles)
    meta = dict(z_bottom=z_bottom, z_top=z_top, px_m=px_m, theta0=theta0,
                gutter_holes=hole_stats[0],
                width_m=2 * np.pi * r_max, height_m=s1 - s0, W=W, H=H, ss=ss)
    return Image.fromarray(img, "RGBA"), meta
