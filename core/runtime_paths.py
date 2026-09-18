# -*- coding: utf-8 -*-
"""运行路径：开发时用项目根目录；PyInstaller打包后用 exe 同级 data 目录（可写）"""
import os
import sys


def is_frozen():
    return getattr(sys, "frozen", False)


def app_root():
    """代码根目录（打包后为 _MEIPASS 临时解压目录，只读）"""
    if is_frozen():
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir():
    """可写数据目录：打包后为 exe所在目录/data；开发时为项目根目录"""
    if is_frozen():
        d = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "data")
    else:
        d = app_root()
    os.makedirs(d, exist_ok=True)
    return d
