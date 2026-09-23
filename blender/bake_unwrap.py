# -*- coding: utf-8 -*-
"""Blender 无头烘焙脚本（由 TowerUnwrap 调用，不单独使用）
思路（借鉴 HB Bake：低模→展UV→烘焙）：
  1. 导入大疆高模 OBJ 分块（带贴图）
  2. 按塔轴剖面生成干净低模圆锥筒（无双层、无黑缝、无破洞）
  3. 柱面 UV 直接铺成展开图布局：u=方位角(门洞居中)，v=斜高
  4. Cycles 漫反射烘焙：高模贴图颜色 → 低模 UV 贴图 = 展开图原图
用法: blender --background --factory-startup --python bake_unwrap.py -- job.json
job.json: {objs:[...], profile:[[z,cx,cy,r],...], z0,z1,px_m,theta0,W,H,out_png}
"""
import bpy, json, math, os, sys


def np_from_image(img):
    import numpy as np
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    return px


def load_job():
    argv = sys.argv[sys.argv.index("--") + 1:]
    with open(argv[0], "r", encoding="utf-8") as f:
        return json.load(f)


def lerp_profile(profile, z):
    """按 z 线性插值 (cx, cy, r)，profile 按 z 升序"""
    zs = [p[0] for p in profile]
    if z <= zs[0]:
        return profile[0][1:]
    if z >= zs[-1]:
        return profile[-1][1:]
    for i in range(len(zs) - 1):
        if zs[i] <= z <= zs[i + 1]:
            t = (z - zs[i]) / (zs[i + 1] - zs[i])
            a, b = profile[i], profile[i + 1]
            return (a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)
    return profile[-1][1:]


def main():
    job = load_job()
    z0, z1 = job["z0"], job["z1"]
    px_m = job["px_m"]
    theta0 = job.get("theta0", 0.0)
    W, H = job["W"], job["H"]
    scale = int(job.get("scale", 2))               # 超采样倍数，烘焙后由调用方降采样
    profile = job["profile"]

    # ---- 1. 清空场景并导入高模 ----
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for p in job["objs"]:
        # 大疆OBJ坐标即最终坐标（z向上），不做轴向转换
        bpy.ops.wm.obj_import(filepath=p, forward_axis='Y', up_axis='Z')
    hi = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert hi, "未导入任何网格"
    # 源贴图关闭插值/多级过滤：烘焙射线微分会让默认过滤沿图集长条方向过度模糊（横向拖影）
    for o in hi:
        for m in o.data.materials:
            if m and m.use_nodes:
                for n in m.node_tree.nodes:
                    if n.type == "TEX_IMAGE":
                        n.interpolation = "Closest"
    print(f"[bake] 高模 {len(hi)} 个对象, "
          f"{sum(len(o.data.polygons) for o in hi)} 面", flush=True)

    # ---- 2. 生成低模圆锥筒（柱面UV=展开图布局） ----
    n_seg = 2048                       # 环向分段（弦误差<1mm，远小于像素）
    z_step = 0.25                      # 环行距（米）
    s0, s1 = 0.0, 0.0
    # 斜高：与内置渲染一致，从 z0 起累计 ds = sqrt(dz² + dr²)
    rows = []
    z = z0
    prev = lerp_profile(profile, z0)
    s = 0.0
    rows.append((z, prev[0], prev[1], prev[2], 0.0))
    while z < z1 - 1e-6:
        zn = min(z + z_step, z1)
        cur = lerp_profile(profile, zn)
        s += math.hypot(zn - z, cur[2] - prev[2])
        rows.append((zn, cur[0], cur[1], cur[2], s))
        z, prev = zn, cur
    s1 = s
    n_r = len(rows)

    verts, faces = [], []
    v_of_row = []
    for (zz, cx, cy, rr, ss) in rows:
        v = ss / s1 if s1 > 0 else 0.0
        v_of_row.append(v)
        for j in range(n_seg):
            th = theta0 - math.pi + 2.0 * math.pi * j / n_seg   # u: 左边缘=θ0-π
            verts.append((cx + rr * math.cos(th), cy + rr * math.sin(th), zz))
    for i in range(n_r - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + (j + 1) % n_seg
            d = (i + 1) * n_seg + j
            faces.append((a, d, c, b))          # 外法线（逆时针从外侧看）
    mesh = bpy.data.meshes.new("unwrap_cyl")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    lo = bpy.data.objects.new("unwrap_cyl", mesh)
    bpy.context.scene.collection.objects.link(lo)

    # UV 层：必须按“面角点”分配，而不是按顶点。
    # 接缝面（j=n_seg-1）若按顶点取 u，会从 0.9995 直接跳回 0.0，
    # 在 UV 空间横跨整张图、盖住所有正常面，烘焙结果就会整行塌缩成同一采样点。
    uv_layer = mesh.uv_layers.new(name="UnwrapUV")
    for poly in mesh.polygons:
        i, j = divmod(poly.index, n_seg)          # from_pydata 保持创建顺序
        u_l, u_r = j / n_seg, (j + 1) / n_seg     # 接缝面正确落在 [0.9995, 1.0]
        v_lo, v_hi = v_of_row[i], v_of_row[i + 1]
        # 面顶点顺序 (a,d,c,b): a=(i,j) d=(i+1,j) c=(i+1,j+1) b=(i,j+1)
        corner_uvs = ((u_l, v_lo), (u_l, v_hi), (u_r, v_hi), (u_r, v_lo))
        for k, li in enumerate(poly.loop_indices):
            uv_layer.data[li].uv = corner_uvs[k]
    mesh.uv_layers.active = uv_layer
    # 法线朝外
    for poly in mesh.polygons:
        poly.use_smooth = False

    # ---- 3. 低模材质 + 烘焙目标图（scale×超采样，治点采样锯齿） ----
    img = bpy.data.images.new("bake_target", width=W * scale, height=H * scale, alpha=True)
    img.generated_color = (0.0, 0.0, 0.0, 0.0)     # 未命中区域保持透明
    mat = bpy.data.materials.new("bake_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    tex_node.select = True
    nt.nodes.active = tex_node                     # 烘焙目标
    lo.data.materials.append(mat)

    # ---- 4. Cycles 烘焙（仅颜色，无光照） ----
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.diffuse_bounces = 0
    sc.cycles.glossy_bounces = 0
    sc.render.bake.use_selected_to_active = True
    sc.render.bake.cage_extrusion = 1.5            # 覆盖拟合面外侧1.5m（爬梯等）
    sc.render.bake.max_ray_distance = 3.0          # 同时覆盖拟合面内侧1.5m
    sc.render.bake.margin = 8
    sc.render.bake.target = "IMAGE_TEXTURES"

    bpy.ops.object.select_all(action="DESELECT")
    for o in hi:
        o.select_set(True)
    lo.select_set(True)
    bpy.context.view_layer.objects.active = lo
    print(f"[bake] 开始烘焙 {W * scale}x{H * scale} (scale={scale}) ...", flush=True)
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    print("[bake] 烘焙完成，保存中...", flush=True)
    # 调试：检查图像缓冲
    px = np_from_image(img)
    print("[bake] 缓冲检查 has_data=%s max=%.3f mean=%.4f" % (
        img.has_data, px.max(), px.mean()), flush=True)

    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    img.save_render(job["out_png"])
    print("[bake] DONE", flush=True)
    # 显式退出：Blender 4.4 后台模式在大型烘焙后的清理阶段偶发崩溃（退出码非 0），
    # 此时输出已完整写盘，直接以 0 退出跳过崩溃的清理流程
    sys.exit(0)


main()
