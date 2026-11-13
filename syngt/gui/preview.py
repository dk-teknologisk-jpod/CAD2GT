"""Live preview panel (renders in a background thread with its own OpenGL context)."""
from __future__ import annotations

import hashlib
import os
import json
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..config import ProjectConfig
from ..visualize import depth_colormap, draw_overlay, instance_colormap, normals_vis


def to_pixmap(img: np.ndarray) -> QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        q = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    elif img.shape[2] == 4:
        q = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        q = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(q.copy())


class ImageView(QGraphicsView):
    """Zoomable / pannable image view. Wheel = zoom at cursor, drag = pan, double-click = fit.
    Hover shows pixel coordinates. The zoom is kept when the image is updated (same size)."""

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._text = self._scene.addText("")
        self._text.setDefaultTextColor(QColor(170, 170, 170))
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setToolTip("Mouse wheel: zoom   drag: pan   double-click: fit to window")
        self._fit = True
        self._size = None
        self.zoom_label = QLabel("")

    def setText(self, t: str):
        self._item.setPixmap(QPixmap())
        self._text.setPlainText(t)
        self._scene.setSceneRect(self._text.boundingRect())
        self.resetTransform()
        self._size = None

    def set_image(self, img: np.ndarray):
        pm = to_pixmap(img)
        self._text.setPlainText("")
        self._item.setPixmap(pm)
        size = (pm.width(), pm.height())
        self._scene.setSceneRect(0, 0, *size)
        if self._fit or size != self._size:
            self._size = size
            self.fit()

    def fit(self):
        if not self._item.pixmap().isNull():
            self.fitInView(self._item, Qt.KeepAspectRatio)
            self._fit = True
            self._update_label()

    def _update_label(self):
        self.zoom_label.setText(f"zoom {self.transform().m11() * 100:.0f}%")

    def wheelEvent(self, e):
        if self._item.pixmap().isNull():
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        s = self.transform().m11() * f
        if 0.02 < s < 40:
            self.scale(f, f)
            self._fit = False
            self._update_label()

    def mouseDoubleClickEvent(self, e):
        self.fit()

    def mouseMoveEvent(self, e):
        super().mouseMoveEvent(e)
        p = self.mapToScene(e.position().toPoint())
        if self._size and 0 <= p.x() < self._size[0] and 0 <= p.y() < self._size[1]:
            self.zoom_label.setText(f"zoom {self.transform().m11() * 100:.0f}%   px ({int(p.x())}, {int(p.y())})")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit:
            self.fit()


def _inputs_signature(cfg: ProjectConfig) -> str:
    """Version of everything on disk a scene reads (model files, texture / photo / distractor folders): the
    preview must change when a file is replaced or a folder gets new content, like a generation run would."""
    from ..textures import file_stamp
    parts = [f"{m.path}:{file_stamp(m.path)}" for m in cfg.enabled_models()]
    for folder in (cfg.scene.ground.texture_dir, cfg.scene.background.image_dir, cfg.objects.distractors.mesh_dir):
        if folder and os.path.isdir(folder):
            for root, _dirs, files in os.walk(folder):
                parts += [f"{f}:{file_stamp(os.path.join(root, f))}" for f in sorted(files)]
    return "|".join(parts)


class PreviewWorker(QObject):
    done = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self):
        super().__init__()
        self.renderer = None
        self.r_key = None
        self.cache_key = None
        self.sampler = None
        self.state = None

    @Slot(object)
    def run(self, req: dict):
        try:
            t0 = time.time()
            cfg = ProjectConfig.model_validate(req["cfg"])
            from ..renderer import Renderer
            from ..scene import SceneSampler
            rk = (cfg.render.msaa_samples, cfg.scene.lighting.shadow_map_size)
            if self.renderer is None or self.r_key != rk:
                self.renderer = Renderer(*rk)
                self.r_key = rk
                self.log.emit(f"OpenGL: {self.renderer.info.get('GL_RENDERER')}")
            ck = hashlib.sha1((json.dumps(req["cfg"], sort_keys=True) + _inputs_signature(cfg)).encode()).hexdigest() \
                + f"|{req['scene']}"
            if ck != self.cache_key or self.state is None:
                # build everything first, then swap: a failure must not leave a sampler of one configuration
                # next to the scene of another
                sampler = SceneSampler(cfg, renderer=self.renderer, log=self.log.emit)
                rng = np.random.default_rng([cfg.generation.seed, req["scene"]])
                state = sampler.sample_scene(req["scene"], rng)
                self.sampler, self.state, self.cache_key = sampler, state, ck
            t1 = time.time()
            v = min(req["view"], len(self.state.views) - 1)
            frames = self.sampler.render_view(self.state, v, np.random.default_rng([cfg.generation.seed, req["scene"], v, 1]),
                                              want_normals=True)
            self.sampler.release_scene_resources()      # meshes of removed parts, old table textures
            t2 = time.time()
            self.done.emit(dict(frames=frames, info=self.state.info, warnings=list(self.sampler.warnings), n_views=len(self.state.views), view=v,
                                class_names=self.sampler.class_names, t_scene=t1 - t0, t_render=t2 - t1,
                                thresholds=dict(min_visib_fract=cfg.output.min_visib_fract,
                                                min_px_visib=cfg.output.min_px_visib,
                                                max_truncation=cfg.output.max_truncation)))
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class PreviewPanel(QWidget):
    request = Signal(object)

    def __init__(self, get_cfg):
        super().__init__()
        self.get_cfg = get_cfg
        self.result = None
        self.busy = False
        self.pending = None
        self.thread = QThread(self)
        self.worker = PreviewWorker()
        self.worker.moveToThread(self.thread)
        self.request.connect(self.worker.run)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        # bound slots, never lambdas: PySide runs a lambda in the *sender's* thread (here the render thread),
        # and touching widgets from there crashes Qt (segfault in QWidgetTextControl)
        self.worker.log.connect(self._log)
        self.thread.start()

        self.btn = QPushButton("Render (F5)")
        self.btn.setToolTip("Re-render the current scene index with the current settings")
        self.btn.clicked.connect(self.render)
        self.next_btn = QPushButton("New scene (F6)")
        self.next_btn.setToolTip("Next scene index -> new objects, poses, lights, background")
        self.next_btn.clicked.connect(self.new_scene)
        self.auto = QCheckBox("auto")
        self.auto.setChecked(True)
        self.auto.setToolTip("Re-render automatically when a setting changes")
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(700)
        self._debounce.timeout.connect(self.render)
        self.scene = QSpinBox()
        self.scene.setRange(0, 10**8)
        self.scene.setPrefix("scene ")
        self.rand = QPushButton("🎲")
        self.rand.setFixedWidth(34)
        self.rand.setToolTip("Random scene index")
        self.rand.clicked.connect(self._random)
        self.view = QSpinBox()
        self.view.setRange(0, 10000)
        self.view.setPrefix("view ")
        self.view.valueChanged.connect(lambda _: self.render())
        self.cam = QComboBox()
        self.cam.currentIndexChanged.connect(self._show)
        self.layer = QComboBox()
        self.layer.addItems(["RGB", "Depth (sensor)", "Depth (GT)", "Instances", "Semantic", "Normals"])
        self.layer.currentIndexChanged.connect(self._show)
        self.overlays = {}
        ov = QHBoxLayout()
        for k, lab, on in [("mask", "mask", False), ("bbox_visib", "bbox", True), ("bbox_amodal", "amodal", False),
                           ("obb", "OBB", False), ("box3d", "3D box", True), ("axes", "axes", False),
                           ("keypoints", "kpts", False), ("labels", "labels", True), ("ignored", "filtered", True)]:
            cb = QCheckBox(lab)
            cb.setChecked(on)
            cb.toggled.connect(self._show)
            self.overlays[k] = cb
            ov.addWidget(cb)
        ov.addStretch(1)
        self._ov_layout = ov
        top = QHBoxLayout()
        self.scene.valueChanged.connect(lambda _: self.render())
        for w in (self.btn, self.next_btn, self.auto, self.scene, self.rand, self.view, self.cam, self.layer):
            top.addWidget(w)
        top.addStretch(1)
        self.img = ImageView()
        ov.addWidget(self.img.zoom_label)
        self.img.setText("Rendering a sample scene …  (F5 = re-render, F6 = new scene)")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 0, 0, 0)
        v.addLayout(top)
        v.addLayout(ov)
        v.addWidget(self.img, 1)
        v.addWidget(self.status)

    @Slot(str)
    def _log(self, s: str):
        self.status.setText(s)

    def shutdown(self):
        """Stop the render thread. A preview that is still simulating/rendering finishes first (seconds);
        destroying a running QThread would abort the whole process."""
        self.pending = None
        self.thread.quit()
        if not self.thread.wait(30000):
            import os
            os._exit(0)   # settings are already saved; never hang or core-dump on exit

    def settings_changed(self):
        if self.auto.isChecked():
            self._debounce.start()

    def new_scene(self):
        self.scene.setValue(self.scene.value() + 1)  # valueChanged -> render

    def _random(self):
        self.scene.setValue(int(np.random.randint(0, 10**6)))

    def render(self):
        cfg = self.get_cfg()
        if not cfg.enabled_models():
            self.status.setText("Add at least one CAD model on the Objects page.")
            return
        req = {"cfg": cfg.to_dict(), "scene": self.scene.value(), "view": self.view.value()}
        if self.busy:
            self.pending = req
            return
        self.busy = True
        self.btn.setEnabled(False)
        self.status.setText("rendering …")
        self.request.emit(req)

    def _next(self):
        self.busy = False
        self.btn.setEnabled(True)
        if self.pending is not None:
            req, self.pending = self.pending, None
            self.busy = True
            self.btn.setEnabled(False)
            self.request.emit(req)

    def _failed(self, msg):
        self.status.setText("<span style='color:#e55'>Preview failed:</span> " + msg.splitlines()[0])
        self.status.setToolTip(msg)
        self._next()

    def _done(self, res):
        self.result = res
        cur = self.cam.currentIndex()
        self.cam.blockSignals(True)
        self.cam.clear()
        for f in res["frames"]:
            self.cam.addItem(f.cam_name)
        self.cam.setCurrentIndex(max(0, min(cur, len(res["frames"]) - 1)))
        self.cam.blockSignals(False)
        self.view.blockSignals(True)
        self.view.setMaximum(max(0, res["n_views"] - 1))
        self.view.setValue(res["view"])
        self.view.blockSignals(False)
        self._show()
        self._next()

    def _show(self):
        res = self.result
        if not res or not res["frames"]:
            return
        f = res["frames"][max(0, self.cam.currentIndex())]
        layer = self.layer.currentText()
        if layer == "RGB":
            base = f.rgb[..., :3]
        elif layer == "Depth (sensor)":
            base = depth_colormap(f.depth if f.depth is not None else f.depth_gt)
        elif layer == "Depth (GT)":
            base = depth_colormap(f.depth_gt)
        elif layer == "Instances":
            base = instance_colormap(f.inst)
        elif layer == "Semantic":
            base = instance_colormap(f.semantic)
        else:
            base = normals_vis(f.normals) if f.normals is not None else f.rgb[..., :3]
        layers = {k: cb.isChecked() for k, cb in self.overlays.items()}
        img = draw_overlay(base, f.annotations, res["class_names"], layers, K=f.K, dist=f.dist, inst_mask=f.inst,
                           thresholds=res["thresholds"])
        self.img.set_image(img)
        n_keep = sum(1 for a in f.annotations if a["visib_fract"] >= res["thresholds"]["min_visib_fract"]
                     and a["px_count_visib"] >= res["thresholds"]["min_px_visib"]
                     and a["truncation"] <= res["thresholds"]["max_truncation"])
        info = res["info"]
        warn = "".join(f"<br><span style='color:#e55'><b>Warning:</b> {w}</span>" for w in res.get("warnings", []))
        self.status.setText(
            f"{f.cam_name}: {f.rgb.shape[1]}x{f.rgb.shape[0]}, fx={f.K[0, 0]:.1f}  |  objects placed {info['n_placed']}/"
            f"{info['n_requested']} ({info['placement_mode']}, sim steps {info['sim_steps']}, settled {info['settled']}, "
            f"{info['placement_time_s']}s)  |  annotations in view {len(f.annotations)}, kept by filters {n_keep}  |  "
            f"bg: {f.background_mode}, ground: {info.get('ground_mode', '')}  |  scene {res['t_scene']:.2f}s, "
            f"render {res['t_render']:.2f}s" + warn)
