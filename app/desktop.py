# -*- coding: utf-8 -*-
"""桌面版入口：后台启动本地服务 + 原生桌面窗口（pywebview/Edge WebView2，
失败回退 PySide6 WebEngine，再失败回退系统浏览器）
PyInstaller 打包入口；开发时也可直接 python app/desktop.py 运行
"""
import sys, os, threading, socket, traceback

if getattr(sys, "frozen", False):
    sys.path.insert(0, sys._MEIPASS)
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_server(port, timeout=30):
    import time, urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    port = _free_port()
    url = f"http://127.0.0.1:{port}/"

    import uvicorn
    from app.server import app
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait_server(port)

    try:
        # 首选：pywebview 原生窗口（使用系统 Edge WebView2，体积小）
        import webview
        from core.runtime_paths import app_root

        class _JsApi:
            """暴露给前端 JS 的原生能力（pywebview 里 <a download> 不会触发系统下载）"""
            def download_file(self, name):
                import shutil
                from core.runtime_paths import data_dir
                name = os.path.basename(name or "")
                src = os.path.join(data_dir(), "output", "app", name)
                if not name or not os.path.exists(src):
                    return {"error": "文件不存在: " + (name or "?")}
                win = webview.windows[0]
                try:
                    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                    res = win.create_file_dialog(
                        webview.SAVE_DIALOG,
                        directory=desktop if os.path.isdir(desktop) else "",
                        save_filename=name)
                except Exception:
                    res = win.create_file_dialog(webview.SAVE_DIALOG,
                                                 save_filename=name)
                if not res:
                    return {"canceled": True}
                dst = res if isinstance(res, str) else res[0]
                shutil.copyfile(src, dst)
                return {"saved": dst}

        icon = os.path.join(app_root(), "app", "icon.ico")
        webview.create_window("风电混塔表面展开工具", url,
                              width=1360, height=900, min_size=(1024, 700),
                              js_api=_JsApi())
        print("[启动] 原生桌面窗口已打开（pywebview/WebView2）", flush=True)
        webview.start(icon=icon if os.path.exists(icon) else None)
        server.should_exit = True
        return
    except Exception:
        traceback.print_exc()

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtWebEngineWidgets import QWebEngineView
        from PySide6.QtCore import QUrl
        qapp = QApplication(sys.argv)
        win = QMainWindow()
        win.setWindowTitle("风电混塔表面展开工具")
        win.resize(1320, 880)
        view = QWebEngineView()
        view.load(QUrl(url))
        win.setCentralWidget(view)
        win.show()
        qapp.exec()
        server.should_exit = True
    except Exception:
        # 回退：系统浏览器
        import webbrowser, time
        print("=" * 56)
        print("  风电混塔表面展开工具 已启动")
        print(f"  界面地址: {url}")
        print("  浏览器将自动打开；使用期间请勿关闭本窗口")
        print("=" * 56, flush=True)
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.should_exit = True


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # 打包后 ProcessPool 子进程必需
    try:
        main()
    except Exception:
        try:
            from core.runtime_paths import data_dir
            log = os.path.join(data_dir(), "app_error.log")
        except Exception:
            log = "app_error.log"
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n===== 启动失败 =====\n" + traceback.format_exc())
        raise
