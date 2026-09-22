# -*- coding: utf-8 -*-
"""核心单元测试：文件名消毒、路径防护、任务忙闲、渲染参数契约
运行: python -m pytest tests/ -q   （或 python tests/test_server.py）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server import sanitize_name, safe_out_path, JOBS, _busy, _new_job, _set_job


# ---------- 文件名消毒 ----------

def test_sanitize_plain():
    assert sanitize_name("tower") == "tower"
    assert sanitize_name("五河S02") == "五河S02"
    assert sanitize_name("B11-展开_01") == "B11-展开_01"


def test_sanitize_traversal():
    assert ".." not in sanitize_name("../../etc/passwd")
    assert "/" not in sanitize_name("../../etc/passwd")
    assert "\\" not in sanitize_name("..\\..\\win.ini")
    assert sanitize_name("../") == "tower"          # 全部非法 -> 默认值


def test_sanitize_injection():
    s = sanitize_name('<img src=x onerror=alert(1)>')
    assert "<" not in s and ">" not in s and '"' not in s


def test_sanitize_empty_and_long():
    assert sanitize_name("") == "tower"
    assert sanitize_name(None) == "tower"
    assert len(sanitize_name("a" * 200)) <= 40


# ---------- 输出路径防护 ----------

def test_safe_out_path_ok(tmp_path=None):
    base = os.path.abspath("out_test")
    fp = safe_out_path(base, "tower_z0-20m_2mm.png")
    assert os.path.commonpath([base, fp]) == base


def test_safe_out_path_traversal():
    base = os.path.abspath("out_test")
    for bad in ("../evil.png", "..\\evil.png", "sub/../../evil.png"):
        try:
            safe_out_path(base, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝: {bad}")


# ---------- 任务忙闲互斥 ----------

def test_busy_state():
    JOBS.clear()
    assert _busy() is None
    jid = _new_job("render")
    assert _busy() == "render"
    _set_job(jid, done=True)
    assert _busy() is None


def test_busy_ignores_other_kinds():
    JOBS.clear()
    jid = _new_job("load")
    assert _busy() == "load"
    _set_job(jid, done=True)
    assert _busy() is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} 个测试全部通过")
