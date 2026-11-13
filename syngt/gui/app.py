"""GUI entry point. Shows a splash screen before the heavy libraries (trimesh, scipy, mujoco, moderngl)
are imported, so a cold start never looks like a hang."""
from __future__ import annotations

import signal
import sys


def _splash(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
    from PySide6.QtWidgets import QSplashScreen
    pm = QPixmap(520, 200)
    pm.fill(QColor(32, 36, 44))
    p = QPainter(pm)
    p.setPen(QColor(235, 235, 235))
    f = QFont()
    f.setPointSize(20)
    f.setBold(True)
    p.setFont(f)
    from .icon import draw_icon
    p.drawPixmap(40, 52, draw_icon(96))
    p.drawText(pm.rect().adjusted(120, -30, 0, 0), Qt.AlignCenter, "syngt")
    f.setPointSize(10)
    f.setBold(False)
    p.setFont(f)
    p.setPen(QColor(170, 170, 170))
    p.drawText(pm.rect().adjusted(120, 40, 0, 0), Qt.AlignCenter, "synthetic ground-truth generator")
    p.end()
    s = QSplashScreen(pm)
    s.show()
    s.showMessage("loading libraries (mesh, physics, rendering) …", Qt.AlignBottom | Qt.AlignHCenter, QColor(200, 200, 200))
    app.processEvents()
    return s


def user_dir():
    """Per-user folder for session/crash files: %APPDATA%\\syngt on Windows, ~/.config/syngt on Linux."""
    import os
    from pathlib import Path
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    d = base / "syngt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _crash_log():
    """On a hard crash (segfault in a native library) dump the Python stacks of all threads to the terminal
    and to crash.log in user_dir(), so the cause can be found."""
    import faulthandler
    faulthandler.enable(sys.stderr, all_threads=True)
    try:
        p = user_dir() / "crash.log"
        if p.exists() and p.stat().st_size > 0:   # keep the trace of the last crash (a restart must not wipe it)
            import time
            p.replace(p.with_name(time.strftime("crash_%Y%m%d_%H%M%S.log", time.localtime(p.stat().st_mtime))))
        f = open(p, "w")
        faulthandler.enable(f, all_threads=True)   # replaces the stderr handler; echo the path on start
        print(f"(crash traces go to {p})", file=sys.stderr)
        return f
    except OSError:
        return None


def run() -> int:
    _CRASH_FILE = _crash_log()  # noqa: F841  keep the file open for the whole session
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("syngt")
    app.setStyle("Fusion")
    if sys.platform == "win32":     # own taskbar entry and icon instead of the generic Python one
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("syngt.app")
        except Exception:
            pass
    from .icon import app_icon
    app.setWindowIcon(app_icon())   # window title bar, task switcher, dock / taskbar - and all dialogs
    splash = _splash(app)
    # Ctrl+C in the terminal closes the app cleanly (Qt otherwise only sees it on the next Python call)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QTimer()
    tick.start(300)
    tick.timeout.connect(lambda: None)

    from .main_window import MainWindow  # heavy imports happen here, behind the splash
    from .widgets import WheelGuard
    guard = WheelGuard(app)             # the mouse wheel scrolls pages, it never changes a setting
    app.installEventFilter(guard)
    splash.showMessage("building window …", Qt.AlignBottom | Qt.AlignHCenter)
    app.processEvents()
    w = MainWindow()
    w.show()
    splash.finish(w)
    rc = app.exec()
    w.preview.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(run())
