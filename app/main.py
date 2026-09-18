# -*- coding: utf-8 -*-
"""风电混塔表面展开工具 - 本地服务入口
用法: python app/main.py [--host H] [--port P]  (也读环境变量 HOST/PORT，默认 127.0.0.1:7100)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _arg(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def main():
    import uvicorn
    host = _arg("--host", os.environ.get("HOST", "127.0.0.1"))
    port = int(_arg("--port", os.environ.get("PORT", "7100")))
    from app.server import app
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
