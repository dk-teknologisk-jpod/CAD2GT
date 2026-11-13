"""Sim-to-real helpers in the UI: colour palettes, clean background plates, match the look of real photos."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..project import Project
from .widgets import Binder

PALETTE_PRESETS = {
    "LEGO common": ["#A0A5A9", "#6C6E68", "#F4F4F4", "#1B2A34", "#237841", "#C91A09", "#0055BF", "#F2CD37"],
    "LEGO greys/white/black": ["#A0A5A9", "#6C6E68", "#F4F4F4", "#1B2A34"],
    "LEGO light grey/white": ["#A0A5A9", "#A0A5A9", "#F4F4F4", "#F4F4F4", "#6C6E68"],
    "Metal / black plastic": ["#B4B6B8", "#8E9093", "#2A2A2A", "#151515"],
    "Neutral (greys)": ["#202020", "#505050", "#808080", "#B0B0B0", "#E0E0E0"],
}


def hex_to_rgb(h: str):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def rgb_to_hex(c) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, x)) * 255)):02X}" for x in c[:3])


class PaletteEdit(QWidget):
    """Hex colour list bound to MaterialConfig.palette, with swatches and presets."""

    def __init__(self, binder: Binder, mat, mode_attr: str = "mode"):
        super().__init__()
        self.binder, self.mat = binder, mat
        self.mode_attr = mode_attr  # field switched to "palette" when colours are entered
        self.edit = QLineEdit(", ".join(rgb_to_hex(c) for c in mat.palette))
        self.edit.setPlaceholderText("#A0A5A9, #F4F4F4, …  (one colour is drawn per instance)")
        self.edit.editingFinished.connect(self._set)
        self.sw = QLabel()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.edit)
        h = QHBoxLayout()
        h.addWidget(self.sw, 1)
        for name in PALETTE_PRESETS:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self._preset(n))
            h.addWidget(b)
        v.addLayout(h)
        self._swatches()

    def _preset(self, name):
        self.edit.setText(", ".join(PALETTE_PRESETS[name]))
        self._set()

    def _set(self):
        cols = []
        for tok in self.edit.text().replace(";", ",").split(","):
            tok = tok.strip()
            if len(tok.lstrip("#")) == 6:
                try:
                    cols.append(hex_to_rgb(tok))
                except ValueError:
                    pass
        if cols == self.mat.palette:
            return
        with self.binder.project.batch("palette"):
            self.mat.palette = cols
            # entering colours means "use them": switch the mode unless it already draws from the palette
            if cols and getattr(self.mat, self.mode_attr) not in ("palette", "mixed", "shadow_catcher"):
                setattr(self.mat, self.mode_attr, "palette")
        self._swatches()

    def _swatches(self):
        self.sw.setText("".join(f"<span style='background:{rgb_to_hex(c)}'>&nbsp;&nbsp;&nbsp;&nbsp;</span>&nbsp;"
                                for c in self.mat.palette) or "<i>no colours</i>")


def build_plates_dialog(parent, project: Project) -> bool:
    """Pick folders of annotated real photos -> clean plates -> set as background photos."""
    cfg = project.cfg
    dlg = QDialog(parent)
    dlg.setWindowTitle("Build clean background plates from real photos")
    dlg.resize(760, 520)
    lst = QListWidget()
    out = QLineEdit(str(Path(cfg.scene.background.image_dir) if cfg.scene.background.image_dir
                        else project.root / "backgrounds" / "workcell_plates"))
    log = QPlainTextEdit()
    log.setReadOnly(True)

    def add():
        d = QFileDialog.getExistingDirectory(dlg, "Folder with real images (+ YOLO .txt / COCO json labels)")
        if d:
            lst.addItem(d)

    def pick_out():
        d = QFileDialog.getExistingDirectory(dlg, "Output folder for plates", out.text())
        if d:
            out.setText(d)
    b_add, b_out = QPushButton("Add folder…"), QPushButton("…")
    b_add.clicked.connect(add)
    b_out.clicked.connect(pick_out)
    info = QLabel("Groups photos taken from the same static camera pose, masks the annotated objects (YOLO boxes / "
                  "polygons or COCO json next to the images) and takes the per-pixel median -> the empty workcell. "
                  "Needs ≥5 photos per camera pose with the parts at different positions. <b>Review the plates</b>: "
                  "unlabelled objects that never moved stay in. Best results: just take 5-20 photos of the empty "
                  "workspace with your camera and put them in the folder directly.")
    info.setWordWrap(True)
    bb = QDialogButtonBox()
    run = bb.addButton("Build plates", QDialogButtonBox.AcceptRole)
    bb.addButton(QDialogButtonBox.Close)
    bb.rejected.connect(dlg.reject)
    v = QVBoxLayout(dlg)
    v.addWidget(info)
    v.addWidget(QLabel("Folders with annotated real images:"))
    v.addWidget(lst)
    v.addWidget(b_add)
    h = QHBoxLayout()
    h.addWidget(QLabel("Output:"))
    h.addWidget(out, 1)
    h.addWidget(b_out)
    v.addLayout(h)
    v.addWidget(log)
    v.addWidget(bb)
    done = {"ok": False}

    def go():
        folders = [lst.item(i).text() for i in range(lst.count())]
        if not folders:
            return
        from ..plates import build_plates
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            files = build_plates(folders, out.text(), log=lambda s: (log.appendPlainText(s), QApplication.processEvents()))
        except Exception as e:
            log.appendPlainText(f"failed: {e}")
            files = []
        finally:
            QApplication.restoreOverrideCursor()
        log.appendPlainText(f"\n{len(files)} clean plate(s) written to {out.text()}")
        if files and QMessageBox.question(dlg, "Use plates", "Use these plates as background photos with an invisible, "
                                          "shadow-catching table (Look: realistic workcell)?") == QMessageBox.Yes:
            def use(c):
                c.scene.background.image_dir = out.text()
                c.scene.background.mode = "image_folder"
                c.scene.ground.texture_mode = "shadow_catcher"
            project.apply("Background: clean plates", use)
            done["ok"] = True
    run.clicked.connect(go)
    dlg.exec()
    return done["ok"]


def table_colours_dialog(parent, project: Project) -> bool:
    """Background option: measure the work-surface colours of real photos -> ground colour palette."""
    d = QFileDialog.getExistingDirectory(parent, "Folder with real photos of the workspace (labels are masked)")
    if not d:
        return False
    from ..lookprofile import apply_surface_colors, measure_surface_colors
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        pal = measure_surface_colors([d], log=lambda s: None)
    except Exception as e:
        QApplication.restoreOverrideCursor()
        QMessageBox.warning(parent, "Table colours", str(e))
        return False
    QApplication.restoreOverrideCursor()
    sw = " ".join(f"<span style='background:{rgb_to_hex(p['color'])}'>&nbsp;&nbsp;&nbsp;&nbsp;</span> "
                  f"{p['weight'] * 100:.0f}%" for p in pal)
    if QMessageBox.question(parent, "Table colours", f"Measured surface colours:<br><br>{sw}<br><br>"
                            "Use them as table colours?") != QMessageBox.Yes:
        return False
    msg = []
    project.apply("Table colours (measured)", lambda c: msg.append(apply_surface_colors(c, pal)))
    QMessageBox.information(parent, "Table colours", msg[0])
    return True


def match_look_dialog(parent, project: Project) -> bool:
    cfg = project.cfg
    d = QFileDialog.getExistingDirectory(parent, "Folder with REAL images from your camera")
    if not d:
        return False
    from ..appearance import apply_to_config, compare_table, folder_stats, synthetic_stats
    QApplication.setOverrideCursor(Qt.WaitCursor)
    try:
        real = folder_stats(d)
        synth = synthetic_stats(cfg)
    except Exception as e:
        QApplication.restoreOverrideCursor()
        QMessageBox.warning(parent, "Match look", str(e))
        return False
    QApplication.restoreOverrideCursor()
    table = compare_table(real, synth)
    if QMessageBox.question(parent, "Match look to real photos",
                            f"Measured {int(real['_n']['mean'])} real images vs 8 synthetic frames:\n\n{table}\n\n"
                            "Adjust the sensor settings (exposure, noise, white balance, saturation, contrast, "
                            "blur) to match?\n(Scene colours are not changed here - use a 'Look' profile / "
                            "background photos for that.)") != QMessageBox.Yes:
        return False
    log = []
    project.apply("Match look to real photos", lambda c: log.extend(apply_to_config(c, real, synth)))
    QMessageBox.information(parent, "Match look", "Changed:\n• " + "\n• ".join(log))
    return True
