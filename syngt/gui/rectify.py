"""Phone photo -> metric, fronto-parallel, tileable ground texture."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget)

from .. import textures as tx
from .preview import ImageView, to_pixmap


class PointCanvas(QWidget):
    """Shows an image; click to place 4 corner points, drag to adjust, right-click to delete."""

    def __init__(self, on_change):
        super().__init__()
        self.img = None
        self.pm = None
        self.pts = []
        self.drag = -1
        self.on_change = on_change
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

    def set_image(self, img):
        self.img = img
        self.pm = to_pixmap(img)
        self.pts = []
        self.update()

    def _xf(self):
        if self.pm is None:
            return 1.0, 0.0, 0.0
        s = min(self.width() / self.pm.width(), self.height() / self.pm.height())
        ox = (self.width() - self.pm.width() * s) / 2
        oy = (self.height() - self.pm.height() * s) / 2
        return s, ox, oy

    def _to_img(self, p):
        s, ox, oy = self._xf()
        return np.array([(p.x() - ox) / s, (p.y() - oy) / s])

    def _to_w(self, q):
        s, ox, oy = self._xf()
        return QPointF(q[0] * s + ox, q[1] * s + oy)

    def _nearest(self, p, tol=15):
        best, bd = -1, tol
        for i, q in enumerate(self.pts):
            w = self._to_w(q)
            d = ((w.x() - p.x()) ** 2 + (w.y() - p.y()) ** 2) ** 0.5
            if d < bd:
                best, bd = i, d
        return best

    def mousePressEvent(self, e):
        if self.pm is None:
            return
        p = e.position()
        i = self._nearest(p)
        if e.button() == Qt.RightButton:
            if i >= 0:
                self.pts.pop(i)
                self.on_change()
                self.update()
            return
        if i >= 0:
            self.drag = i
        elif len(self.pts) < 4:
            self.pts.append(self._to_img(p))
            self.drag = len(self.pts) - 1
            self.on_change()
        self.update()

    def mouseMoveEvent(self, e):
        if self.drag >= 0:
            self.pts[self.drag] = self._to_img(e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drag >= 0:
            self.drag = -1
            self.on_change()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self.pm is None:
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignCenter, "Load a photo of the support surface (e.g. table) with a\n"
                                                    "known rectangle on it (A4 sheet, cutting mat, table edges).")
            return
        s, ox, oy = self._xf()
        p.drawPixmap(int(ox), int(oy), int(self.pm.width() * s), int(self.pm.height() * s), self.pm)
        p.setRenderHint(QPainter.Antialiasing)
        ordered = order_points(self.pts) if len(self.pts) == 4 else self.pts
        if len(ordered) >= 2:
            p.setPen(QPen(QColor(0, 255, 120), 2))
            p.drawPolygon(QPolygonF([self._to_w(q) for q in ordered]))
        for i, q in enumerate(ordered):
            w = self._to_w(q)
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.drawEllipse(w, 6, 6)
            p.drawText(w + QPointF(8, -8), ["TL", "TR", "BR", "BL"][i] if len(ordered) == 4 else str(i + 1))


def order_points(pts):
    P = np.asarray(pts, float)
    s = P.sum(1)
    d = P[:, 1] - P[:, 0]
    return [P[np.argmin(s)], P[np.argmin(d)], P[np.argmax(s)], P[np.argmax(d)]]


class RectifyDialog(QDialog):
    def __init__(self, parent=None, target_path_edit=None):
        super().__init__(parent)
        self.setWindowTitle("Phone photo → metric ground texture")
        self.resize(1300, 800)
        self.target = target_path_edit
        self.src_path = ""
        self.result = None
        self.canvas = PointCanvas(self._update_preview)
        self.prev = ImageView()
        self.prev.setText("rectified preview")
        self.w_mm = QDoubleSpinBox()
        self.w_mm.setRange(1, 100000)
        self.w_mm.setValue(297)
        self.w_mm.setSuffix(" mm")
        self.h_mm = QDoubleSpinBox()
        self.h_mm.setRange(1, 100000)
        self.h_mm.setValue(210)
        self.h_mm.setSuffix(" mm")
        self.ppm = QDoubleSpinBox()
        self.ppm.setRange(0.1, 50)
        self.ppm.setValue(2.0)
        self.ppm.setSuffix(" px/mm")
        self.flat = QCheckBox("Flatten illumination (remove shading/vignetting -> albedo-like)")
        self.flat.setChecked(True)
        self.seam = QCheckBox("Make tileable (cross-fade seams)")
        self.seam.setChecked(True)
        self.inset = QDoubleSpinBox()
        self.inset.setRange(0, 0.45)
        self.inset.setSingleStep(0.02)
        self.inset.setValue(0.0)
        for w in (self.w_mm, self.h_mm, self.ppm, self.inset):
            w.valueChanged.connect(self._update_preview)
        self.flat.toggled.connect(self._update_preview)
        self.seam.toggled.connect(self._update_preview)
        self.name = QLineEdit("table_texture")
        b_load = QPushButton("Load photo…")
        b_load.clicked.connect(self._load)
        b_a4 = QPushButton("A4")
        b_a4.clicked.connect(lambda: (self.w_mm.setValue(297), self.h_mm.setValue(210)))
        b_reset = QPushButton("Clear points")
        b_reset.clicked.connect(lambda: (self.canvas.pts.clear(), self.canvas.update(), self._update_preview()))
        b_save = QPushButton("Save texture to folder…")
        b_save.clicked.connect(self._save)
        form = QFormLayout()
        form.addRow("Rectangle width (TL→TR)", self.w_mm)
        form.addRow("Rectangle height (TL→BL)", self.h_mm)
        form.addRow("Texture resolution", self.ppm)
        form.addRow("Crop inset (fraction)", self.inset)
        form.addRow(self.flat)
        form.addRow(self.seam)
        form.addRow("Texture name", self.name)
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.addWidget(QLabel("<b>1.</b> Load a photo  <b>2.</b> click the 4 corners of a rectangle of known size lying on the "
                            "surface  <b>3.</b> enter its size  <b>4.</b> save. The texture is saved with a .json sidecar "
                            "(metric size) and is tiled with true scale on the ground plane."))
        sv.itemAt(0).widget().setWordWrap(True)
        sv.addWidget(b_load)
        sv.addLayout(form)
        sv.addWidget(QWidget())
        h = QHBoxLayout()
        h.addWidget(b_a4)
        h.addWidget(b_reset)
        sv.addLayout(h)
        sv.addWidget(self.prev, 1)
        sv.addWidget(b_save)
        sp = QSplitter()
        sp.addWidget(self.canvas)
        sp.addWidget(side)
        sp.setStretchFactor(0, 3)
        sp.setStretchFactor(1, 2)
        QVBoxLayout(self).addWidget(sp)

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Photo", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.heic)")
        if not p:
            return
        try:
            img = tx.load_image_rgb(p, 4000)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.src_path = p
        self.name.setText(Path(p).stem + "_rect")
        self.canvas.set_image(img)
        self._update_preview()

    def _update_preview(self, *_):
        if self.canvas.img is None or len(self.canvas.pts) != 4:
            self.result = None
            return
        pts = np.asarray(order_points(self.canvas.pts), np.float32)
        img = self.canvas.img
        if self.flat.isChecked():
            img = tx.flatten_illumination(img)
        self.result = None      # a failed rectification must not leave the previous result to be saved
        try:
            self.result = tx.rectify_photo(img, pts, self.w_mm.value(), self.h_mm.value(), self.ppm.value(),
                                           self.seam.isChecked(), self.inset.value())
        except Exception as e:
            self.prev.set_image(np.zeros((8, 8, 3), np.uint8))
            self.setWindowTitle(f"Rectify - cannot rectify with these values: {e}")
            return
        tile = np.tile(self.result, (2, 2, 1)) if self.seam.isChecked() else self.result
        self.prev.set_image(tile)

    def _save(self):
        if self.result is None:
            QMessageBox.information(self, "Rectify", "Place 4 points first.")
            return
        start = self.target.edit.text() if self.target is not None else ""
        d = QFileDialog.getExistingDirectory(self, "Texture folder", start)
        if not d:
            return
        inset = self.inset.value()
        w_mm = self.w_mm.value() * (1 - 2 * inset)
        h_mm = self.h_mm.value() * (1 - 2 * inset)
        out = Path(d) / (self.name.text() + ".png")
        tx.save_texture(self.result, str(out), w_mm, h_mm, self.src_path)
        if self.target is not None:
            self.target.set_path(d)
        QMessageBox.information(self, "Saved", f"Saved {out}\n(tile = {w_mm:.0f} x {h_mm:.0f} mm)\n\n"
                                               "Set the ground texture mode to 'image folder' or 'mixed' to use it.")
