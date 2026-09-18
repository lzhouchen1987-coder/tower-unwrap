# -*- coding: utf-8 -*-
"""展开图后处理：小孔洞填补 + 低频光照均化（去孤岛色差，效果近似PS匀色）"""
import sys
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def suppress_sky_pixels(rgb, alpha, px_m, dilate_iter=2):
    """识别天空污染的亮青色像素（无人机照片在塔顶/板缝处拍到天空，被贴图烘焙进模型），
    将其标记为空洞，由后续补洞流程用周边混凝土纹理填补。
    两阶段判据，避免误伤亮蓝油漆：
    1) 种子：很亮且明显偏蓝（L>125, b<110）——亮蓝油漆 L≤115 够不到；
    2) 扩张：偏蓝候选像素只有在"邻域(~15cm)以混凝土/种子为主"时才认作天空——
       蓝色油漆带内部的亮蓝补丁四周都是油漆，不会被误伤。"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    valid = alpha >= 128
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(A - 128.0, B - 128.0)
    seed = valid & (L > 125) & (B < 110) & (A < 140)
    if not seed.any():
        return rgb, alpha
    base = ((C < 18) | seed).astype(np.float32)              # 混凝土 + 已知天空
    nbr = _masked_lowpass(base, valid.astype(np.float32), 0.15, px_m)
    cand = valid & (L > 110) & (B < 118) & (A < 142) & (C > 18)
    sky = cand & (nbr > 0.4)
    if not sky.any():
        return rgb, alpha
    sky = cv2.dilate(sky.astype(np.uint8), np.ones((3, 3), np.uint8),
                     iterations=dilate_iter).astype(bool) & valid
    out_a = alpha.copy()
    out_a[sky] = 0
    return rgb, out_a


def fill_small_holes(rgb, alpha, max_dist_px=40):
    """填补被塔面包围的小孔洞（距有效像素<max_dist_px的透明区）"""
    from scipy.ndimage import distance_transform_edt
    hole = alpha < 128
    if not hole.any():
        return rgb, alpha
    dist, inds = distance_transform_edt(hole, return_indices=True)
    small = hole & (dist <= max_dist_px)
    if small.any():
        rgb = rgb.copy()
        a = alpha.copy()
        rgb[small] = rgb[inds[0][small], inds[1][small]]
        a[small] = 255
        return rgb, a
    return rgb, alpha


def push_pull_fill(rgb, alpha, max_levels=14, edge_max_px=None):
    """推拉金字塔补洞：空洞区域由周边颜色平滑扩散填充（适合大面积/带状空洞）。
    内部洞（不与图像边界相连）全部填补；边界连通的洞只补距有效像素 edge_max_px
    以内的部分（塔身边缘缺数据区补纹理，画布外大片空白保持透明）。
    返回 (填充后rgb, 填充后alpha)。有效像素保持原值。"""
    hole = alpha < 128
    if not hole.any():
        return rgb, alpha
    # 只补内部洞：与图像边界连通的空洞属于画布外区域，不填
    n, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=4)
    border_ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    internal = hole & ~np.isin(lab, list(border_ids))
    if edge_max_px:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(hole)
        edge_zone = hole & ~internal & (dist <= edge_max_px)
        internal = internal | edge_zone
    if not internal.any():
        return rgb, alpha
    img = rgb.astype(np.float32)
    w = (alpha >= 128).astype(np.float32)
    pyr = [(img, w)]
    while min(pyr[-1][0].shape[:2]) > 24 and len(pyr) < max_levels:
        i2 = cv2.pyrDown(pyr[-1][0] * pyr[-1][1][..., None])
        w2 = cv2.pyrDown(pyr[-1][1])
        pyr.append((i2, w2))
    i, w = pyr[-1]
    cur = i / np.maximum(w[..., None], 1e-6)
    for i, w in reversed(pyr[:-1]):
        up = cv2.pyrUp(cur, dstsize=(i.shape[1], i.shape[0]))
        own = i / np.maximum(w[..., None], 1e-6)
        a = np.clip(w, 0, 1)[..., None]
        cur = own * a + up * (1 - a)
    filled = np.clip(cur, 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[internal] = filled[internal]
    out_a = alpha.copy()
    out_a[internal] = 255
    return out, out_a


def _masked_lowpass(channel, mask, sigma_m, px_m, return_den=False):
    """掩膜加权低频场：降采样+归一化高斯，返回全分辨率低频图(float32)"""
    ds = max(1, int(round((sigma_m / px_m) / 64)))
    small_c = cv2.resize(channel, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA).astype(np.float32)
    small_m = cv2.resize(mask, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
    k = int(2 * round((sigma_m / px_m) / ds / 2) + 1)
    k = max(3, k | 1)
    num = cv2.GaussianBlur(small_c * small_m, (k, k), 0)
    den = cv2.GaussianBlur(small_m, (k, k), 0)
    low = num / np.maximum(den, 1e-3)
    low = cv2.resize(low, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_LINEAR)
    if return_den:
        den = cv2.resize(den, (channel.shape[1], channel.shape[0]), interpolation=cv2.INTER_LINEAR)
        return low, den
    return low


def flatten_lab(rgb, alpha, sigma_m=0.8, px_m=0.002, l_strength=0.9, c_strength=0.85, meds=None,
                chroma_lo=12.0, chroma_hi=26.0, contrast_strength=0.9,
                shade_keep=0.6, clarity=0.7, gcontrast=1.1):
    """Lab空间低频均化：L通道做"均值+局部对比度"仿射匹配（去曝光差和雾霾差导致的补丁感）；
    a/b通道把低频色度偏差拉回中位（去蓝/黄大补丁）。锈迹、裂缝等高频细节按等比保留。
    meds可传入全图Lab中位数（分块处理时保持基准一致）。
    高色度像素（红/蓝油漆标识带等，经~5cm低通后C仍>chroma_hi）既不参与低频场估计，
    也不被均化——保护真实标识颜色；C在chroma_lo~chroma_hi之间线性过渡。"""
    mask = (alpha >= 128).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    out = lab.copy()
    C = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    sigma_c = max(3.0, 0.05 / px_m)                          # ~5cm 色度低通
    C_low = cv2.GaussianBlur(C, (0, 0), sigma_c)             # 细描边被平滑掉，宽带保留
    pw = np.clip((C_low - chroma_lo) / max(chroma_hi - chroma_lo, 1e-6), 0, 1)   # 1=完全保护
    free = (1.0 - pw).astype(np.float32)                     # 可被均化的权重
    field_mask = mask * free                                 # 低频场只用近中性像素估计
    if meds is None:
        m = alpha >= 128
        meds = [float(np.median(lab[..., c][m])) for c in range(3)]
    # L：三尺度分解 = 保留自然明暗(>4m，如向阳/背阳渐变) + 去除照片补丁(0.35~4m) + 纹理增强(<0.35m)
    low_l = _masked_lowpass(lab[..., 0], field_mask, sigma_m, px_m)
    low_xl = _masked_lowpass(lab[..., 0], field_mask, 4.0, px_m)
    res_l = lab[..., 0] - low_l
    var_l = _masked_lowpass(np.clip(res_l * res_l, 0, 4000), field_mask, sigma_m, px_m)
    std_l = np.sqrt(np.maximum(var_l, 1.0))
    core = (alpha >= 128) & (free > 0.5)
    tstd = float(np.median(std_l[core])) if core.any() else 8.0
    cf = np.clip(tstd / std_l, 0.55, 1.9)
    cf = 1.0 + (cf - 1.0) * contrast_strength
    new_l = meds[0] + (low_xl - meds[0]) * shade_keep + res_l * cf * l_strength \
        + (low_l - meds[0]) * (1.0 - l_strength)
    # 质感增强（清晰度）：放大 2~3cm 尺度的细部，治"白茫茫"的雾霾感
    if clarity > 0:
        sig_d = max(2.0, 0.025 / px_m)
        det = new_l - cv2.GaussianBlur(new_l, (0, 0), sig_d)
        new_l = new_l + det * clarity
    # 全局对比度轻拉（以中位亮度为锚点）
    if gcontrast != 1.0:
        new_l = meds[0] + (new_l - meds[0]) * gcontrast
    # 保护区（油漆标识带）：向"带内大尺度均值"对齐，消除带内照片曝光补丁；
    # 只用保护类像素估计带内场——小面积饱和特征（锈迹）局部场≈自身，自动不受影响。
    # 两个尺度要拉开差距（0.5m vs 2m），中间尺度的补丁才能被捕获
    prot_mask = mask * pw
    sigma_pb = max(sigma_m * 1.5, 0.5)
    sigma_xl = 2.0                                          # 带内平均尺度（米）
    lowp_l, _ = _masked_lowpass(lab[..., 0], prot_mask, sigma_pb, px_m, return_den=True)
    lowx_l, denx = _masked_lowpass(lab[..., 0], prot_mask, sigma_xl, px_m, return_den=True)
    trust = np.clip(denx / 0.05, 0, 1)                      # 邻域内保护类像素太少则不动
    prot_l = lab[..., 0] + (lowx_l - lowp_l) * l_strength * trust
    out[..., 0] = pw * prot_l + free * new_l
    # a/b：混凝土加性偏移回中性；保护区向带内大尺度色对齐
    for c in (1, 2):
        low_c = _masked_lowpass(lab[..., c], field_mask, sigma_m, px_m)
        shift = (meds[c] - low_c) * c_strength * free        # 保护区不偏向中性
        lowp_c, _ = _masked_lowpass(lab[..., c], prot_mask, sigma_pb, px_m, return_den=True)
        lowx_c, _ = _masked_lowpass(lab[..., c], prot_mask, sigma_xl, px_m, return_den=True)
        shift = shift + pw * (lowx_c - lowp_c) * c_strength * trust
        out[..., c] = lab[..., c] + shift
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[alpha < 128] = lab[alpha < 128].astype(np.uint8)   # 画布外保持原色（黑），避免色偏
    return cv2.cvtColor(out, cv2.COLOR_Lab2RGB)


def transfer_grain(rgb_orig, rgb_filled, alpha, strength=0.85, grain_sigma=10, edge_max_px=None):
    """把有效区的高频颗粒（混凝土质感）按最近邻迁移到补洞区：
    残差 = 原图 - 掩膜归一化低频；洞像素 += 最近有效像素残差"""
    hole = alpha < 128
    if not hole.any():
        return rgb_filled
    n, lab = cv2.connectedComponents(hole.astype(np.uint8), connectivity=4)
    border_ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    internal = hole & ~np.isin(lab, list(border_ids))
    from scipy.ndimage import distance_transform_edt
    if edge_max_px:
        dist0 = distance_transform_edt(hole)
        internal = internal | (hole & ~internal & (dist0 <= edge_max_px))
    if not internal.any():
        return rgb_filled
    w = (alpha >= 128).astype(np.float32)
    k = int(2 * round(grain_sigma * 3) + 1) | 1
    num = cv2.GaussianBlur(rgb_orig.astype(np.float32) * w[..., None], (k, k), 0)
    den = cv2.GaussianBlur(w, (k, k), 0)
    low = num / np.maximum(den[..., None], 1e-3)
    res = rgb_orig.astype(np.float32) - low          # 有效区高频残差
    # 残差限幅：防止填补区迁入黑斑/异物颗粒（取有效区残差的稳健分布边界）
    rv = res[w > 0]
    if rv.size:
        glo = float(np.percentile(rv, 1)) * 1.5
        ghi = float(np.percentile(rv, 99)) * 1.5
        res = np.clip(res, glo, ghi)
    _, inds = distance_transform_edt(internal, return_indices=True)
    grain = res[inds[0][internal], inds[1][internal]]
    out = rgb_filled.astype(np.float32)
    out[internal] += grain * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def suppress_blue_rims(rgb, alpha, px_m, thresh=10.0, strength=0.85, neutral=0.4, chroma_protect=20):
    """抑制天空污染的青色描边（只作用于近中性色混凝土，保护油漆标识/锈迹等饱和色）：
    1) 低色度像素中"比邻域更蓝"的Lab b离群值软压回邻域（去青色描边）
    2) a/b 中位数向中性128回收 neutral 比例，按色度加权（混凝土去蓝，油漆不褪色）"""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    valid = alpha >= 128
    C = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)     # 色度
    sigma_c = max(3.0, 0.05 / px_m)                            # ~5cm 色度低通：细描边不算"宽带"
    C_low = cv2.GaussianBlur(C, (0, 0), sigma_c)
    protect = C_low > chroma_protect                          # 高饱和宽色带（油漆标识）不动
    k = max(15, int(round(0.035 / px_m)) | 1)                  # ~3.5cm 中值窗
    b8 = np.clip(lab[..., 2], 0, 255).astype(np.uint8)
    med = cv2.medianBlur(b8, k).astype(np.float32)
    dev = med - lab[..., 2]                                    # >0：比邻域更蓝
    corr = np.clip(dev - thresh, 0, None) * strength
    corr[protect] = 0
    lab[..., 2] += corr
    if neutral > 0:
        w = np.clip((chroma_protect - C_low) / chroma_protect, 0, 1)   # C_low=0→1，≥阈值→0
        for c in (1, 2):
            m = lab[..., c][valid & ~protect]
            if m.size:
                cur = float(np.median(m))
                lab[..., c] += (128.0 - cur) * neutral * w
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    out[~valid] = rgb[~valid]
    return out


def _global_medians_lab(rgb, alpha, step=8):
    """降采样估计全图 Lab 三通道中位数（分块处理时作为统一基准，避免块间色阶）"""
    small = rgb[::step, ::step]
    m = alpha[::step, ::step] >= 128
    if not m.any():
        return [128.0, 128.0, 128.0]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)
    return [float(np.median(lab[..., c][m])) for c in range(3)]


def process_pipeline(rgb, alpha, px_m, tile_rows=8192, sigma_m=0.35,
                     strength=0.85, clarity=0.7, on_progress=None):
    """大图后处理流水线（分块省内存）：天空污染填补→补洞→颗粒迁移→Lab均化，按行分块+重叠拼接。
    strength：匀色强度0~1（照片补丁/色差去除力度）；clarity：质感增强0~1（清晰度/对比度）。
    返回 (处理后的rgb, 处理后的alpha)"""
    H, W = alpha.shape
    meds = _global_medians_lab(rgb, alpha)
    sigma_px = sigma_m / px_m
    halo = int(min(4096, max(2048, 3 * sigma_px + 64, 2.2 * 2.0 / px_m)))
    out = np.zeros_like(rgb)
    out_a = alpha.copy()
    n_tiles = (H + tile_rows - 1) // tile_rows
    for ti, r0 in enumerate(range(0, H, tile_rows)):
        r1 = min(H, r0 + tile_rows)
        s0, s1 = max(0, r0 - halo), min(H, r1 + halo)
        t_rgb, t_a = rgb[s0:s1], alpha[s0:s1]
        t_rgb, t_a = suppress_sky_pixels(t_rgb, t_a, px_m)   # 天空污染像素标记为空洞
        edge_px = int(1.5 / px_m)                            # 边缘缺数据区填补深度1.5m
        filled, a2 = push_pull_fill(t_rgb, t_a, edge_max_px=edge_px)
        filled = transfer_grain(t_rgb, filled, t_a, edge_max_px=edge_px)
        flat = flatten_lab(filled, a2, sigma_m=sigma_m, px_m=px_m,
                           l_strength=strength, c_strength=strength,
                           contrast_strength=0.9 * strength,
                           shade_keep=1.0 - 0.5 * strength,
                           clarity=clarity, gcontrast=1.0 + 0.15 * clarity,
                           meds=meds)
        flat = suppress_blue_rims(flat, a2, px_m)
        keep = slice(r0 - s0, r1 - s0)
        out[r0:r1] = flat[keep]
        out_a[r0:r1] = a2[keep]
        del filled, flat, a2, t_rgb, t_a
        if on_progress:
            on_progress(ti + 1, n_tiles)
    return out, out_a


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else r"output\yiyang08_band_2mm.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".png", "_flat.png")
    im = Image.open(src)
    arr = np.asarray(im)
    rgb, alpha = arr[..., :3], arr[..., 3]
    print("分块后处理（补洞+颗粒迁移+Lab均化）...", flush=True)
    flat, alpha2 = process_pipeline(rgb, alpha, px_m=0.002,
                                    on_progress=lambda i, n: print(f"  块 {i}/{n}", flush=True))
    out = np.dstack([flat, alpha2])
    Image.fromarray(out, "RGBA").save(dst)
    print("已保存", dst)
    # 处理前后色偏统计对比
    for name, im3 in (("处理前", rgb), ("处理后", flat)):
        d = im3[..., 2].astype(np.int16) - im3[..., 0].astype(np.int16)
        dv = d[alpha >= 128]
        print(f"{name}: B-R 中位={np.median(dv):.0f} 90分位={np.percentile(dv,90):.0f} 最大={dv.max()}", flush=True)
    # 对比小图
    w, h = flat.shape[1], flat.shape[0]
    Image.fromarray(flat).resize((1400, int(h * 1400 / w))).save(dst.replace(".png", "_view.jpg"), quality=88)
