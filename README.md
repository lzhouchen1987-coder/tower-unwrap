# TowerUnwrap 风电混塔表面展开工具

把大疆无人机实景三维模型（OBJ，带照片纹理）的**风电混塔塔身表面展开成平面图**，
用于塔筒外观检测、缺陷标注与存档。

- 展开起点可选：**门洞自动放到展开图水平正中**
- 展开高度自定义：从塔底到塔顶，可整塔一张，也可按高度分多张
- 输出带透明背景的 PNG，可直接拖入 Photoshop 拼接修饰
- 双渲染引擎：
  - **内置快速渲染**（推荐）：自研射线投射 + 纹理重采样，速度快
  - **Blender 烘焙**：低模圆锥筒 + 柱面 UV + Cycles 烘焙，照片拼缝色块更少
    （需本机安装 Blender 4.x，思路借鉴自
    [blender-agent-bake](https://github.com/zhaimingyou/blender-agent-bake)）
- 一键匀色：去除无人机照片之间的曝光/白平衡色差、天空青斑，同时保留
  塔身自然明暗与混凝土纹理质感，油漆标识带颜色自动保护
- 原生桌面窗口（pywebview / Edge WebView2），无需打开浏览器

## 运行环境

- Windows 10 / 11（64 位）
- 可选：Blender 4.x（仅"Blender 烘焙"引擎需要，
  自动查找 `C:\Program Files\Blender Foundation\` 等常见安装目录，
  也可设置环境变量 `TOWER_BLENDER` 指定 blender.exe 路径）

## 使用步骤

1. 双击 `TowerUnwrap.exe`，等待软件窗口打开。
2. 选择大疆三维模型目录（包含各 Block 分块的 `terra_obj` 目录）。
   首次加载需重建纹理缓存（约 5~10 分钟），之后秒开。
3. 在剖面图上设置展开高度范围。
4. 在预览图上点选门洞位置（放到展开图中间）。
5. 设置分辨率（推荐 2mm/像素）、分张高度、渲染引擎。
6. 点击开始渲染。结果输出在 `data\output\app\` 目录。

## 从源码运行

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app\desktop.py
```

## 打包

```bash
.venv\Scripts\python -m PyInstaller tower_unwrap.spec --noconfirm
```

产物在 `dist\TowerUnwrap\`。

## 目录结构

```
app/        FastAPI 服务 + 前端界面 + 桌面入口
core/       展开渲染核心（剖面拟合、射线投射、纹理重采样、Blender烘焙桥接）
blender/    Blender 无头烘焙脚本（低模圆锥筒 + 柱面UV + Cycles 烘焙）
scripts/    后处理管线（匀色、补洞、质感增强）
```

## 致谢

Blender 烘焙引擎的思路（高模 → 低模 → 展 UV → 烘焙）参考了
[zhaimingyou/blender-agent-bake](https://github.com/zhaimingyou/blender-agent-bake)。
