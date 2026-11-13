"""'Quick setup' tab: the whole workflow on one page.

Parts -> how they lie -> where the camera is -> background -> camera look -> fine-tune -> dataset.
Everything here edits the same project as the detailed tabs; bulk choices (profiles) report what they changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..assets import MESH_EXTS, guess_file_unit, list_mesh_files, model_extent
from ..config import ObjectModel
from ..profiles import BACKGROUND_PROFILES, LAYOUT_PROFILES, VIEW_PROFILES, current_background_profile
from ..project import Project
from ..viewpoints import views_per_scene
from .widgets import Binder, muted_style, scroll_page

MESH_FILTER = "CAD / mesh (" + " ".join(f"*{e}" for e in sorted(MESH_EXTS)) + ")"
FORMAT_LABELS = [("native", "native (always)"), ("bop", "BOP (6D pose)"), ("coco", "COCO"),
                 ("coco_bb", "COCO boxes for RT-DETR (train/ + validation/)"), ("yolo", "YOLO boxes"),
                 ("yolo_obb", "YOLO oriented boxes"), ("yolo_seg", "YOLO segmentation"), ("yolo_pose", "YOLO pose")]


class RangeSlider(QWidget):
    """Two linked sliders (min / max) with a value read-out, in display units."""

    def __init__(self, lo: float, hi: float, step: float, unit: str, getter: Callable[[], Tuple[float, float]],
                 setter: Callable[[float, float], None], scale: float = 1.0, decimals: int = 0):
        super().__init__()
        self.lo, self.hi, self.step, self.unit, self.scale, self.dec = lo, hi, step, unit, scale, decimals
        self.getter, self.setter = getter, setter
        n = int(round((hi - lo) / step))
        self.a, self.b = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        for s in (self.a, self.b):
            s.setRange(0, n)
            s.setMinimumWidth(160)
        self.lab = QLabel()
        self.lab.setMinimumWidth(130)
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QLabel("from"), 0, 0)
        g.addWidget(self.a, 0, 1)
        g.addWidget(QLabel("to"), 1, 0)
        g.addWidget(self.b, 1, 1)
        g.addWidget(self.lab, 0, 2, 2, 1)
        self.refresh()
        self.a.valueChanged.connect(lambda _: self._moved(True))
        self.b.valueChanged.connect(lambda _: self._moved(False))

    def _to_val(self, i):
        return self.lo + i * self.step

    def _to_idx(self, v):
        return int(round((min(max(v, self.lo), self.hi) - self.lo) / self.step))

    def refresh(self):
        lo, hi = self.getter()
        for s, v in ((self.a, lo), (self.b, hi)):
            s.blockSignals(True)
            s.setValue(self._to_idx(v * self.scale))
            s.blockSignals(False)
        self._label()

    def _label(self):
        a, b = self._to_val(self.a.value()), self._to_val(self.b.value())
        self.lab.setText(f"<b>{a:.{self.dec}f} – {b:.{self.dec}f} {self.unit}</b>")

    def _moved(self, first: bool):
        if self.a.value() > self.b.value():
            (self.b if first else self.a).blockSignals(True)
            (self.b if first else self.a).setValue((self.a if first else self.b).value())
            (self.b if first else self.a).blockSignals(False)
        self.setter(self._to_val(self.a.value()) / self.scale, self._to_val(self.b.value()) / self.scale)
        self._label()


class SingleSlider(QWidget):
    def __init__(self, lo, hi, step, unit, getter, setter, decimals=0):
        super().__init__()
        self.lo, self.step, self.unit, self.dec = lo, step, unit, decimals
        self.getter, self.setter = getter, setter
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(0, int(round((hi - lo) / step)))
        self.s.setMinimumWidth(160)
        self.lab = QLabel()
        self.lab.setMinimumWidth(130)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.s, 1)
        h.addWidget(self.lab)
        self.refresh()
        self.s.valueChanged.connect(self._moved)

    def refresh(self):
        self.s.blockSignals(True)
        self.s.setValue(int(round((self.getter() - self.lo) / self.step)))
        self.s.blockSignals(False)
        self.lab.setText(f"<b>{self.getter():.{self.dec}f} {self.unit}</b>")

    def _moved(self, i):
        self.setter(self.lo + i * self.step)
        self.lab.setText(f"<b>{self.getter():.{self.dec}f} {self.unit}</b>")


def _profile_group(title: str, profiles, on_pick, selected: Optional[str] = None) -> QGroupBox:
    """Compact profile chooser: drop-down + the description of the highlighted profile."""
    g = QGroupBox(title)
    v = QVBoxLayout(g)
    cb = QComboBox()
    cb.addItem("— choose a profile —", None)
    for key, p in profiles.items():
        cb.addItem(p.title, key)
        cb.setItemData(cb.count() - 1, p.description, Qt.ToolTipRole)
    desc = QLabel()
    desc.setWordWrap(True)
    desc.setMinimumHeight(48)
    desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    desc.setStyleSheet(muted_style())

    def show(i):
        k = cb.itemData(i)
        desc.setText(profiles[k].description if k else "Pick one to set all related options at once; "
                                                         "fine-tune below or on the detailed tabs.")
    if selected in profiles:
        cb.setCurrentIndex(list(profiles).index(selected) + 1)
    show(cb.currentIndex())
    cb.highlighted.connect(show)  # hovering an entry explains it

    def picked(i):
        show(i)
        if cb.itemData(i):
            on_pick(cb.itemData(i))
    cb.activated.connect(picked)
    v.addWidget(cb)
    v.addWidget(desc)
    return g


class QuickSetupPage(QWidget):
    def __init__(self, project: Project, report: Callable[[str, list], None] = lambda label, changed: None,
                 rebuild: Callable[[], None] = lambda: None):
        """report(label, changed_settings) is called after a bulk choice (profile ...) so the window can tell the
        user exactly what it changed; rebuild() asks the window to build this page anew (e.g. new look profile)."""
        super().__init__()
        self.project, self.report, self.rebuild = project, report, rebuild
        self.b = Binder(project)
        cfg = self.cfg = project.cfg
        v, p = cfg.viewpoints, cfg.placement

        intro = QLabel("<b>Quick setup</b> – work from top to bottom, the preview on the right follows. "
                       "Everything stays editable in detail on the other tabs. <b>Ctrl+Z</b> takes back any change.")
        intro.setWordWrap(True)

        g_layout = _profile_group("2 · How are the parts lying?", LAYOUT_PROFILES,
                                  lambda k: self.report("layout", project.apply_layout_profile(k)),
                                  project.choices.get("layout"))
        g_view = _profile_group("3 · Where is the camera?", VIEW_PROFILES,
                                lambda k: self.report("camera position", project.apply_view_profile(k)),
                                project.choices.get("view"))
        top = QWidget()
        h = QHBoxLayout(top)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(g_layout, 1)
        h.addWidget(g_view, 1)

        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        self.info.setStyleSheet("background: palette(base); padding: 8px; border: 1px solid palette(mid);")
        g_info = QGroupBox("What the camera will see")
        QVBoxLayout(g_info).addWidget(self.info)

        QVBoxLayout(self).addWidget(scroll_page(intro, self._parts_group(), top, self._background_group(),
                                                self._look_group(), self._fine_group(v, p), g_info,
                                                self._dataset_group()))
        self._info()

    def on_project_changed(self, kind: str) -> None:
        """Called by the main window after every edit while this page is visible."""
        self._info()

    # ------------------------------------------------------------------ 1 · parts
    def _parts_group(self) -> QGroupBox:
        g = QGroupBox("1 · Parts (CAD models)")
        v = QVBoxLayout(g)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        models = self.cfg.objects.models
        per_part = self.cfg.objects.count_mode == "per_model"
        tip_min = "Fewest pieces of this part in a scene."
        tip_max = ("Most pieces of this part in a scene.\nFor every scene a random number between the two is "
                   "drawn (e.g. 2 and 5 → 2, 3, 4 or 5 pieces). Same value twice = always that many.")
        if not per_part:
            tip_min = tip_max = ("Not used: 'Count mode' on the Objects tab is set to 'total', so the number of "
                                 "pieces is drawn for the whole scene, not per part.")
        if models:
            for col, text, tip in ((0, "use", "Untick to leave a part out without removing it"), (1, "part", ""),
                                   (2, "colour", "Click a colour to change it"),
                                   (3, "pieces per scene:  at least", tip_min), (4, "at most", tip_max)):
                lab = QLabel(text)
                lab.setStyleSheet(muted_style())
                lab.setToolTip(tip)
                grid.addWidget(lab, 0, col)
        for r, m in enumerate(models, 1):
            ext = model_extent(m.path, m.unit) if m.path else None
            name = QLabel(f"{m.class_name}" + (f"  <span style='color:gray'>{ext * 1000:.0f} mm</span>"
                                                               if ext else "  <span style='color:#e55'>file?</span>"))
            name.setToolTip(m.path)
            rm = QPushButton("✕")
            rm.setFixedWidth(26)
            rm.setToolTip("Remove this part from the project")
            rm.clicked.connect(lambda _=False, m=m: self._remove(m))
            colour = self._colour_cell(m)
            spins = [self.b.ispin(m.count, attr, 0, 500, tip=tip) for attr, tip in (("min", tip_min), ("max", tip_max))]

            def used(on, name=name, spins=spins):    # a part that is left out: its row is greyed
                name.setEnabled(on)
                for sp in spins:
                    sp.setEnabled(on and per_part)
            grid.addWidget(self.b.check(m, "enabled", on_change=used,
                                        tip="Use this part in the scenes (untick to leave it out without "
                                            "removing it)"), r, 0)
            grid.addWidget(name, r, 1)
            grid.addWidget(colour, r, 2)
            grid.addWidget(spins[0], r, 3)
            grid.addWidget(spins[1], r, 4)
            used(m.enabled)
            grid.addWidget(rm, r, 5)
        v.addLayout(grid)
        if not models:
            v.addWidget(QLabel("No parts yet – add the CAD files of the parts you want to detect "
                               "(STL, OBJ, PLY, GLB, STEP …)."))
        row = QHBoxLayout()
        for text, fn in (("Add CAD files…", self._add_files), ("Add folder…", self._add_folder)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        hint = QLabel("The two numbers are how many pieces of the part lie in a scene (a random number from … to …). "
                      "Colour: files without their own colour (STL, STEP) get the colour picked here. "
                      "More options (palette, random colours, units, symmetry): Objects tab.")
        hint.setWordWrap(True)
        hint.setStyleSheet(muted_style())
        row.addWidget(hint, 1)
        v.addLayout(row)
        return g

    def _colour_cell(self, m: ObjectModel) -> QWidget:
        mat = m.material
        if mat.mode in ("palette", "random_color"):   # set on the Objects tab: show it, do not hide it
            btn = QPushButton("palette" if mat.mode == "palette" else "random")
            btn.setToolTip("This part uses a colour palette / random colours (Objects tab). "
                           "Click to give it one fixed colour instead.")
            btn.setFixedWidth(90)

            def one_colour():
                with self.project.batch("one colour"):
                    mat.mode = "fixed_color"
            btn.clicked.connect(one_colour)
            return btn
        return self.b.color(mat, "color", tip="Colour of this part (click to change)",
                            on_pick=lambda: setattr(mat, "mode", "fixed_color"))

    def _add_paths(self, paths) -> None:
        if not paths:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with self.project.batch(f"add {len(paths)} part(s)"):
                for p in paths:
                    nid = max([m.class_id for m in self.cfg.objects.models], default=0) + 1
                    self.cfg.objects.models.append(ObjectModel(path=str(Path(p).resolve()), name=Path(p).stem,
                                                               class_id=nid, unit=guess_file_unit(p)))
        finally:
            QApplication.restoreOverrideCursor()

    def _add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add CAD files", "", MESH_FILTER + ";;All (*)")
        self._add_paths(ps)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add all CAD files from folder")
        if d:
            self._add_paths(list_mesh_files(d))

    def _remove(self, m: ObjectModel):
        with self.project.batch(f"remove part {m.name}"):
            self.cfg.objects.models.remove(m)

    # ------------------------------------------------------------------ 4 · background
    def _background_group(self) -> QGroupBox:
        from .realism import build_plates_dialog, rgb_to_hex, table_colours_dialog
        project, cfg = self.project, self.cfg
        g = QGroupBox("4 · Background")
        bl = QVBoxLayout(g)
        key = current_background_profile(cfg)
        combo = QComboBox()
        for k, pr in BACKGROUND_PROFILES.items():
            combo.addItem(pr.title, k)
        combo.setCurrentIndex(list(BACKGROUND_PROFILES).index(key))
        desc = QLabel(BACKGROUND_PROFILES[key].description)
        desc.setWordWrap(True)
        desc.setStyleSheet(muted_style())
        bl.addWidget(combo)
        bl.addWidget(desc)

        def picked(i):
            k = combo.itemData(i)
            changed = project.apply_background_profile(k)
            if k == "table_colours" and not cfg.scene.ground.palette:
                table_colours_dialog(self, project)
            self.report("background", changed)
        combo.activated.connect(picked)

        if key == "photos":
            row = QHBoxLayout()
            b_pl = QPushButton("Build from annotated photos…")
            b_pl.setToolTip("Remove the labelled parts from series of real photos -> clean plates of the empty workspace")
            b_pl.clicked.connect(lambda: build_plates_dialog(self, project))
            row.addWidget(QLabel("Photo folder:"))
            row.addWidget(self.b.path(cfg.scene.background, "image_dir", "dir",
                                      tip="Photos of your real, EMPTY workspace (or clean plates)"), 1)
            row.addWidget(b_pl)
            bl.addLayout(row)
        elif key == "table_colours":
            row = QHBoxLayout()
            sw = QLabel("".join(f"<span style='background:{rgb_to_hex(c)}'>&nbsp;&nbsp;&nbsp;&nbsp;</span>&nbsp;"
                                for c in cfg.scene.ground.palette) or "<i>not measured yet</i>")
            b_tc = QPushButton("Measure from real photos…")
            b_tc.clicked.connect(lambda: table_colours_dialog(self, project))
            row.addWidget(QLabel("Table colours:"))
            row.addWidget(sw, 1)
            row.addWidget(b_tc)
            bl.addLayout(row)
        return g

    # ------------------------------------------------------------------ 5 · camera look
    def _look_group(self) -> QGroupBox:
        from ..lookprofile import default_profile, list_profiles, load_profile, set_default_profile
        project, cfg = self.project, self.cfg
        g = QGroupBox("5 · Camera colour && lighting")
        cl = QVBoxLayout(g)
        combo = QComboBox()
        combo.addItem("— choose —", None)
        dflt = default_profile()
        cur = 0
        for p in list_profiles():
            try:
                pr = load_profile(p)
            except Exception:
                continue
            combo.addItem(pr.get("name", Path(p).stem) + ("  (default)" if dflt and Path(p) == Path(dflt) else ""), p)
            combo.setItemData(combo.count() - 1, f"{pr.get('n_images', '?')} real images, created {pr.get('created', '')}",
                              Qt.ToolTipRole)
            if cfg.look_profile and cfg.look_profile.replace("\\", "/").rsplit("/", 1)[-1] == Path(p).name:
                cur = combo.count() - 1
        combo.addItem("Randomised camera & lighting (domain randomisation)", "__random__")
        combo.setCurrentIndex(cur)
        light = QCheckBox("incl. lighting")
        light.setChecked(True)
        light.setToolTip("Also set the lighting the profile's exposure was tuned with (recommended)")

        def picked(i):
            p = combo.itemData(i)
            if p == "__random__":
                self.report("camera look", project.randomise_look())
            elif p:
                self.report("camera look", project.apply_look(p, lighting=light.isChecked()))
        combo.activated.connect(picked)

        def create():
            from .lookdialog import CreateLookDialog
            CreateLookDialog(self, project).exec()
            self.rebuild()          # new profile in the list

        def make_default():
            p = combo.currentData()
            if p and p != "__random__":
                set_default_profile(p)
                self.rebuild()
        b_ml = QPushButton("Extract from real photos…")
        b_ml.setToolTip("Measure brightness, colour cast, saturation, contrast, noise and sharpness of real images "
                        "and auto-tune a reusable profile")
        b_ml.clicked.connect(create)
        b_def = QPushButton("Set as default")
        b_def.setToolTip("Apply this profile automatically to new projects")
        b_def.clicked.connect(make_default)
        row = QHBoxLayout()
        row.addWidget(combo, 1)
        row.addWidget(light)
        row.addWidget(b_ml)
        row.addWidget(b_def)
        desc = QLabel("Brightness, colour cast, saturation, contrast, noise, sharpness and lighting of a real "
                      "camera, extracted from its photos. Independent of the background.")
        desc.setWordWrap(True)
        desc.setStyleSheet(muted_style())
        cl.addLayout(row)
        cl.addWidget(desc)
        return g

    # ------------------------------------------------------------------ 6 · fine-tune
    def _fine_group(self, v, p) -> QGroupBox:
        project, cfg = self.project, self.cfg
        d = cfg.objects.distractors

        def set_range(obj, attr, label):
            def f(a, b):
                r = getattr(obj, attr)      # in place: other tabs are bound to this very object
                r.min, r.max = float(a), float(b)
                project.touch(label)
            return f

        def set_dist(a, b):
            v.distance_mode = "absolute"    # the slider is in centimetres: it only makes sense as a distance
            v.radius.min, v.radius.max = float(a), float(b)
            project.touch("camera distance")

        def set_roll(x):
            v.roll_deg.min, v.roll_deg.max = -float(x), float(x)
            project.touch("camera roll")

        def set_jitter(x):
            project.set(v, "look_at_jitter", x / 100, "aim jitter")

        def set_distractors(x):
            d.enabled = x > 0
            d.count.max = int(x)
            d.count.min = min(d.count.min, d.count.max)
            project.touch("distractors")

        fine = QGroupBox("6 · Fine-tune")
        fg = QGridLayout(fine)
        rows = [
            ("Camera distance to parts", RangeSlider(5, 200, 1, "cm", lambda: (v.radius.min, v.radius.max),
                                                     set_dist, scale=100)),
            ("Viewing angle above table\n(90° = straight down)", RangeSlider(
                -90, 90, 1, "°", lambda: (v.elevation_deg.min, v.elevation_deg.max),
                set_range(v, "elevation_deg", "viewing angle"))),
            ("Camera roll (tilt around\nthe viewing axis) ±", SingleSlider(0, 180, 1, "°", lambda: v.roll_deg.max,
                                                                         set_roll)),
            ("Aim jitter (std)", SingleSlider(0, 10, 0.5, "cm", lambda: v.look_at_jitter * 100, set_jitter, 1)),
            ("Spread of dropped parts\n(small = pile)", RangeSlider(
                0, 30, 0.5, "cm", lambda: (p.cluster_radius.min, p.cluster_radius.max),
                set_range(p, "cluster_radius", "spread of parts"), scale=100, decimals=1)),
            ("Distractor objects (max)", SingleSlider(0, 20, 1, "", lambda: d.count.max if d.enabled else 0,
                                                      set_distractors)),
            ("Camera views per scene", self.b.ispin(v, "views_per_scene", 1, 1000)),
        ]
        for i, (lab, w) in enumerate(rows):
            l = QLabel(lab)
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            fg.addWidget(l, i, 0)
            if isinstance(w, QSpinBox):
                fg.addWidget(w, i, 1, Qt.AlignLeft)
            else:
                fg.addWidget(w, i, 1)
        return fine

    # ------------------------------------------------------------------ 7 · dataset
    def _dataset_group(self) -> QGroupBox:
        cfg, b = self.cfg, self.b
        g = QGroupBox("7 · Dataset")
        grid = QGridLayout(g)
        grid.setColumnStretch(1, 1)
        rows = [("Output folder", b.path(cfg.output, "output_dir", "dir")),
                ("Number of scenes", b.ispin(cfg.generation, "num_scenes", 1, 10**8)),
                ("Label formats", b.flags(cfg.output, "formats", FORMAT_LABELS, columns=3, locked=("native",)))]
        for i, (lab, w) in enumerate(rows):
            l = QLabel(lab)
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(l, i, 0)
            if isinstance(w, QSpinBox):
                grid.addWidget(w, i, 1, Qt.AlignLeft)
            else:
                grid.addWidget(w, i, 1)
        self.total = QLabel()
        grid.addWidget(self.total, len(rows), 1)
        return g

    # ------------------------------------------------------------------ read-out
    def _info(self):
        cfg, v = self.cfg, self.cfg.viewpoints
        lines = []
        if v.mode == "fixed":
            lines.append("Viewpoint mode <b>fixed</b>: cameras stay at their world poses (distance sliders unused).")
        elif v.distance_mode != "absolute":
            lines.append("Distance mode is <b>fill fraction</b> (Viewpoints tab) – moving the distance slider "
                         "switches back to absolute distances.")
        dmin, dmax = v.radius.min, v.radius.max
        models = [(m.class_name, model_extent(m.path, m.unit)) for m in cfg.enabled_models()]
        for c in cfg.cameras:
            fw = lambda d: (c.width / c.fx * d * 100, c.height / c.fy * d * 100)  # noqa: E731
            a, b = fw(dmin), fw(dmax)
            lines.append(f"<b>{c.name}</b> ({c.width}×{c.height}, fx {c.fx:.0f}px): field of view on the table "
                         f"≈ {a[0]:.0f}×{a[1]:.0f} cm at {dmin * 100:.0f} cm … {b[0]:.0f}×{b[1]:.0f} cm at "
                         f"{dmax * 100:.0f} cm.")
            for name, ext in models:
                if ext:
                    p0, p1 = c.fx * ext / max(dmax, 1e-3), c.fx * ext / max(dmin, 1e-3)
                    flag = " <span style='color:#d80'>(small – consider closer / higher resolution)</span>" if p0 < 30 else ""
                    lines.append(f"&nbsp;&nbsp;• {name} ({ext * 1000:.0f} mm): about <b>{p0:.0f}–{p1:.0f} px</b> "
                                 f"long{flag}")
        self.info.setText("<br>".join(lines) or "Add a part to see how large it appears in the image.")
        n_views = views_per_scene(v)
        n_img = cfg.generation.num_scenes * n_views
        self.total.setText(f"{cfg.generation.num_scenes} scenes × {n_views} views = <b>{n_img} images</b> per camera"
                           + (f" × {len(cfg.cameras)} cameras" if len(cfg.cameras) > 1 else ""))
