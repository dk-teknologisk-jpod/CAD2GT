"""Windows hybrid-graphics helper: make the Python interpreters use the high-performance (NVIDIA) GPU.

OpenGL contexts on Optimus laptops go to the iGPU unless the executable is marked as "High performance"
(per-user setting, same as Windows Settings > System > Display > Graphics). The venv python.exe is only a
launcher for the base interpreter, so both are marked. Takes effect for newly started processes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"


def interpreter_paths() -> List[str]:
    out = set()
    for exe in (sys.executable, getattr(sys, "_base_executable", sys.executable)):
        p = Path(exe)
        for name in ("python.exe", "pythonw.exe"):
            q = p.with_name(name)
            if q.exists():
                out.add(str(q))
    return sorted(out)


def set_gpu_preference(high_performance: bool = True) -> List[str]:
    if os.name != "nt":
        return []
    import winreg
    paths = interpreter_paths()
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, KEY) as k:
        for p in paths:
            if high_performance:
                winreg.SetValueEx(k, p, 0, winreg.REG_SZ, "GpuPreference=2;")
            else:
                try:
                    winreg.DeleteValue(k, p)
                except FileNotFoundError:
                    pass
    return paths


def current_preferences() -> dict:
    if os.name != "nt":
        return {}
    import winreg
    out = {}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY) as k:
            for p in interpreter_paths():
                try:
                    out[p] = winreg.QueryValueEx(k, p)[0]
                except FileNotFoundError:
                    out[p] = None
    except FileNotFoundError:
        pass
    return out
