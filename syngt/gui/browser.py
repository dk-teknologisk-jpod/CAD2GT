"""Dataset browser: inspect generated images with all GT overlays (QA before training)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..visualize import depth_colormap, draw_overlay, instance_colormap
from .preview import ImageView


def _imread(p: Path, flags=cv2.IMREAD_UNCHANGED):
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flags)


class DatasetBrowser(QDialog):
    def __init__(self, parent=None, folder: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Dataset browser")
        self.resize(1400, 850)
        self.root = None
        self.class_names = {}
        self.thresholds = {}
        self.split = QComboBox()
        self.split.addItems(["train", "val", "test"])
        self.split.currentIndexChanged.connect(self._fill)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        self.layer = QComboBox()
        self.layer.addItems(["RGB", "Depth", "Instance mask"])
        self.layer.currentIndexChanged.connect(self._show)
        self.multi = QCheckBox("all cameras of frame")
        self.multi.toggled.connect(self._show)
        self.ov = {}
        ovw = QHBoxLayout()
        for k, lab, on in [("mask", "mask", True), ("bbox_visib", "bbox", True), ("bbox_amodal", "amodal", False),
                           ("obb", "OBB", False), ("box3d", "3D box", True), ("axes", "axes", True),
                           ("keypoints", "kpts", False), ("labels", "labels", True), ("ignored", "filtered", True)]:
            cb = QCheckBox(lab)
            cb.setChecked(on)
            cb.toggled.connect(self._show)
            self.ov[k] = cb
            ovw.addWidget(cb)
        ovw.addStretch(1)
        self.img = ImageView()
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(180)
        b = QPushButton("Open dataset…")
        b.clicked.connect(self._open)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(b)
        lv.addWidget(self.split)
        lv.addWidget(self.list, 1)
        lv.addWidget(self.stats)
        right = QWidget()
        rv = QVBoxLayout(right)
        top = QHBoxLayout()
        top.addWidget(self.layer)
        top.addWidget(self.multi)
        top.addStretch(1)
        rv.addLayout(top)
        rv.addLayout(ovw)
        rv.addWidget(self.img, 1)
        rv.addWidget(self.txt)
        sp = QSplitter()
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setStretchFactor(1, 4)
        QVBoxLayout(self).addWidget(sp)
        if folder and (Path(folder) / "annotations").exists():
            self.load(folder)

    def _open(self):
        d = QFileDialog.getExistingDirectory(self, "Dataset folder")
        if d:
            self.load(d)

    def load(self, folder):
        self.root = Path(folder)
        self.class_names, self.thresholds = {}, {}      # never carry names / filters over from another dataset
        try:
            import yaml
            cfg = yaml.safe_load(open(self.root / "config.yaml", encoding="utf-8"))
            for m in cfg["objects"]["models"]:
                self.class_names.setdefault(m["class_id"], (m.get("name") or "").strip() or Path(m["path"]).stem)
            o = cfg["output"]
            self.thresholds = dict(min_visib_fract=o["min_visib_fract"], min_px_visib=o["min_px_visib"],
                                   max_truncation=o["max_truncation"])
        except Exception:
            pass
        try:
            info = json.load(open(self.root / "dataset_info.json"))
            self.stats.setText(f"images per split: {info.get('splits')}<br>instances per class: {info.get('per_class')}<br>"
                               f"visib. histogram (0..1): {info.get('visib_hist')}")
        except Exception:
            self.stats.setText("")
        self.setWindowTitle(f"Dataset browser - {self.root}")
        self._fill()

    def _fill(self):
        self.list.clear()
        if self.root is None:
            return
        d = self.root / "annotations" / self.split.currentText()
        self.files = sorted(d.glob("*.json")) if d.exists() else []
        for p in self.files:
            self.list.addItem(p.stem)
        if self.files:
            self.list.setCurrentRow(0)

    def _render_one(self, p: Path):
        A = json.load(open(p, encoding="utf-8"))
        files = A["image"]["files"]
        layer = self.layer.currentText()
        inst = None
        if "instance_mask" in files:
            from ..writers import read_label_png
            mp = self.root / files["instance_mask"]
            inst = read_label_png(mp) if mp.exists() else None
        if layer == "Depth" and "depth" in files:
            d = _imread(self.root / files["depth"])
            base = depth_colormap(d.astype(np.float32)) if d is not None else None
        elif layer == "Instance mask" and inst is not None:
            base = instance_colormap(inst)
        else:
            base = None
        if base is None:
            im = _imread(self.root / files["rgb"], cv2.IMREAD_COLOR) if (self.root / files["rgb"]).exists() else None
            if im is None:      # image file missing / unreadable: show a placeholder instead of crashing
                im = np.zeros((A["image"]["height"], A["image"]["width"], 3), np.uint8)
                cv2.putText(im, f"missing: {files['rgb']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
            elif layer in ("Depth", "Instance mask"):
                cv2.putText(im, f"no {layer.lower()} was saved for this dataset - showing the image", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
            base = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        layers = {k: cb.isChecked() for k, cb in self.ov.items()}
        img = draw_overlay(base, A["objects"], self.class_names, layers, K=np.array(A["camera"]["K"]),
                           dist=np.array(A["camera"]["dist"]), inst_mask=inst, thresholds=self.thresholds)
        return img, A

    def _show(self, *_):
        r = self.list.currentRow()
        if self.root is None or r < 0 or r >= len(self.files):
            return
        p = self.files[r]
        img, A = self._render_one(p)
        if self.multi.isChecked():
            prefix = "_".join(p.stem.split("_")[:2]) + "_"
            sibs = sorted(p.parent.glob(prefix + "*.json"))
            imgs = [self._render_one(s)[0] for s in sibs]
            hgt = min(i.shape[0] for i in imgs)
            imgs = [cv2.resize(i, (int(i.shape[1] * hgt / i.shape[0]), hgt)) for i in imgs]
            img = np.hstack(imgs)
        self.img.set_image(img)
        lines = [f"{A['image']['stem']}  camera={A['image']['camera']}  scene={A['image']['scene_id']} view={A['image']['view_id']}",
                 f"K={np.round(np.array(A['camera']['K']), 2).tolist()}  dist={np.round(A['camera']['dist'], 5).tolist()}",
                 f"sensor={ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in A.get('sensor', {}).items()} }"]
        for a in A["objects"]:
            t = np.round(a["cam_t_m2c_mm"], 1).tolist()
            lines.append(f"  #{a['inst_id']:3d} {a['class_name']:16s} visib={a['visib_fract']:.2f} trunc={a['truncation']:.2f} "
                         f"px={a['px_count_visib']:6d} bbox={a['bbox_visib']} t_mm={t}")
        self.txt.setPlainText("\n".join(lines))
