# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：风电混塔表面展开工具（onedir + 桌面窗口）"""
from PyInstaller.utils.hooks import collect_all

o3d_datas, o3d_bins, o3d_hidden = collect_all('open3d')
wv_datas, wv_bins, wv_hidden = collect_all('webview')

a = Analysis(
    ['app/desktop.py'],
    pathex=['.'],
    binaries=o3d_bins + wv_bins,
    datas=[('app/static', 'app/static'), ('app/icon.ico', 'app'),
           ('blender', 'blender')] + o3d_datas + wv_datas,
    hiddenimports=o3d_hidden + wv_hidden + [
        'core.unwrap_core', 'core.runtime_paths', 'core.blender_bake',
        'scripts.post_flatten',
        'webview', 'clr', 'pythonnet', 'clr_loader',
        'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto', 'uvicorn.loops.asyncio',
        'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets', 'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan', 'uvicorn.lifespan.on', 'uvicorn.lifespan.off',
        'h11', 'scipy.ndimage', 'cv2',
    ],
    hookspath=[],
    excludes=['matplotlib', 'pandas', 'notebook', 'PyQt5', 'PyQt6', 'tkinter',
              'IPython', 'jupyter', 'seaborn', 'bokeh',
              'PySide6', 'shiboken6',
              'open3d.ml', 'torch', 'torchvision', 'torchgen', 'xformers', 'triton',
              'onnxruntime', 'transformers', 'sklearn', 'av', 'pyarrow',
              'polars', 'tensorboard', 'caffe2', 'jax', 'flax'],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='TowerUnwrap',
    debug=False,
    strip=False,
    upx=False,
    console=True,
    icon='app/icon.ico',
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False,
    name='TowerUnwrap',
)
