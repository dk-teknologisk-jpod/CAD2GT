"""Main window: a thin shell around a :class:`syngt.project.Project`.

The window owns no settings and no rules. It shows the project, forwards user actions to it and reacts to its
change events (EDIT / BULK / RESET). Generation runs in a separate process (``python -m syngt generate``).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..project import BULK, EDIT, RESET, Project
from ..validation import ERROR, WARNING
from .app import user_dir
from .browser import DatasetBrowser
from .pages import CamerasPage, ObjectsPage, OutputPage, PlacementPage, ScenePage, SensorPage, ViewpointsPage
from .preview import PreviewPanel
from .quick import QuickSetupPage
from .rectify import RectifyDialog

EXAMPLE = Project().root / "examples" / "example_config.yaml"
SETTING_NAMES = {"objects": "parts", "placement": "placement", "scene.ground": "table", "scene.background": "background",
                 "scene.lighting": "lighting", "cameras": "cameras", "viewpoints": "viewpoints", "sensor": "sensor",
                 "render": "renderer", "output": "output", "generation": "generation"}


def summarise_changes(changed: list) -> str:
    """'placement.mode, placement.cluster_radius.min' -> 'placement (2), ...' for the status bar."""
    groups: dict = {}
    for k in changed:
        key = next((v for p, v in SETTING_NAMES.items() if k.startswith(p)), k.split(".")[0])
        groups[key] = groups.get(key, 0) + 1
    return ", ".join(f"{k} ({n})" for k, n in groups.items())


class GeneratePanel(QWidget):
    """Pre-flight checks, the Generate button and the log of the generation process."""

    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win, self.project = win, win.project
        self.proc = None
        self.t0 = 0
        self.checks = QListWidget()
        self.checks.setMaximumHeight(92)
        self.checks.setToolTip("Things that would not work as set up. Double-click an entry to open the tab that fixes it.")
        self.checks.itemDoubleClicked.connect(lambda it: self.win.show_tab(it.data(Qt.UserRole)))
        self.ready = QLabel()
        self.start = QPushButton("▶  Generate dataset")
        self.start.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self.start.clicked.connect(self.run)
        self.cancel = QPushButton("Cancel")
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.stop)
        self.reexp = QPushButton("Re-export formats only")
        self.reexp.setToolTip("Rebuild COCO/YOLO exports from native annotations with the current filters (no rendering)")
        self.reexp.clicked.connect(self.reexport)
        self.open = QPushButton("Open folder")
        self.open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.output_dir()))))
        self.browse = QPushButton("Browse dataset")
        self.browse.clicked.connect(lambda: DatasetBrowser(self, str(self.project.output_dir())).exec())
        self.bar = QProgressBar()
        self.eta = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        h = QHBoxLayout()
        for w in (self.start, self.cancel, self.reexp, self.open, self.browse):
            h.addWidget(w)
        h.addWidget(self.ready, 1)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.checks)
        v.addLayout(h)
        h2 = QHBoxLayout()
        h2.addWidget(self.bar, 1)
        h2.addWidget(self.eta)
        v.addLayout(h2)
        v.addWidget(self.log, 1)

    # ------------------------------------------------------------------ checks
    def refresh_checks(self):
        issues = self.project.issues()
        self.checks.clear()
        icon = {ERROR: "⛔", WARNING: "⚠", "info": "ℹ"}
        for i in issues:
            it = QListWidgetItem(f"{icon.get(i.severity, '')}  {i.message}" + (f"   → {i.where}" if i.where else ""))
            it.setData(Qt.UserRole, i.where)
            self.checks.addItem(it)
        self.checks.setVisible(bool(issues))
        n_err = sum(i.severity == ERROR for i in issues)
        self.ready.setText("<span style='color:#e55'>fix the ⛔ entries above first</span>" if n_err else
                           (f"{len(issues)} note(s) above" if issues else "<span style='color:#3a3'>✓ ready</span>"))

    # ------------------------------------------------------------------ process
    def _launch(self, args):
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(self.project.root))
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        env = self.proc.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._stdout)
        self.proc.readyReadStandardError.connect(self._stderr)
        self.proc.finished.connect(self._finished)
        self.start.setEnabled(False)
        self.reexp.setEnabled(False)
        self.cancel.setEnabled(True)
        self.t0 = time.time()
        self.proc.start(sys.executable, ["-m", "syngt"] + args)

    def run(self):
        project = self.project
        blocking = project.blocking_issues()
        if blocking:
            QMessageBox.warning(self, "Generate", "Cannot generate yet:\n\n• " + "\n• ".join(i.message for i in blocking))
            return
        st = project.dataset_status()
        fresh = mixed = False
        if st.foreign_files:
            if QMessageBox.question(self, "Generate", f"{st.out}\nis not empty and not a syngt dataset. "
                                    "Write the dataset into it anyway?") != QMessageBox.Yes:
                return
        elif st.would_mix:
            # the classic trap: finished scenes are skipped, so old images made with other settings stay in
            shown = "\n".join(f"   • {k}" for k in st.changed[:12]) + ("\n   • …" if len(st.changed) > 12 else "")
            box = QMessageBox(QMessageBox.Question, "Settings changed since this dataset was started",
                              f"{st.out}\nalready holds {st.done_scenes} finished scene(s) that were made with "
                              f"different settings:\n\n{shown}\n\nStart fresh replaces them. Keeping them would mix "
                              f"old and new images in one dataset.", parent=self)
            b_fresh = box.addButton("Start fresh (delete old dataset)", QMessageBox.AcceptRole)
            b_keep = box.addButton("Keep old scenes, add the rest", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(b_fresh)
            box.exec()
            if box.clickedButton() is b_fresh:
                fresh = True
            elif box.clickedButton() is b_keep:
                mixed = True
            else:
                return
        elif st.complete:
            box = QMessageBox(QMessageBox.Question, "Dataset is already complete",
                              f"All {st.total_scenes} scenes in\n{st.out}\nare finished with exactly these settings.",
                              parent=self)
            b_again = box.addButton("Generate again from scratch", QMessageBox.DestructiveRole)
            b_exp = box.addButton("Only re-export formats", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is b_exp:
                return self.reexport()
            if box.clickedButton() is not b_again:
                return
            fresh = True
        cfg_file, out = project.write_run_config()
        self.log.clear()
        self.bar.setRange(0, project.cfg.generation.num_scenes)
        self.bar.setValue(0)
        args = ["generate", str(cfg_file), "--json-progress", "--out", str(out)]
        if fresh:
            args.append("--no-resume")
        elif mixed:
            args.append("--allow-mixed")
        elif st.done_scenes:
            self.log.appendPlainText(f"continuing: {st.done_scenes} of {st.total_scenes} scenes are already done "
                                     f"with the same settings")
        self.log.appendPlainText(f"$ python -m syngt {' '.join(args)}")
        self._launch(args)

    def reexport(self):
        out = self.project.output_dir()
        if not (out / "config.yaml").exists():
            QMessageBox.information(self, "Re-export", "No dataset in the output folder yet.")
            return
        o = self.project.cfg.output
        from ..config import DETECTOR_FORMATS
        fm = [f for f in o.formats if f in DETECTOR_FORMATS]
        if not fm:
            QMessageBox.information(self, "Re-export", "No detector format (COCO / YOLO …) is selected "
                                                       "(Quick setup → Dataset, or the Output tab).")
            return
        args = ["export", str(out), "--formats", *fm, "--min-visib", str(o.min_visib_fract), "--min-px", str(o.min_px_visib),
                "--max-trunc", str(o.max_truncation), "--obb-source", o.obb_source]
        self.log.appendPlainText(f"$ python -m syngt {' '.join(args)}")
        self._launch(args)

    def stop(self):
        if self.proc is not None:
            self.proc.kill()
            self.log.appendPlainText("cancelled (finished scenes are kept; resume later). Run 'Re-export' to build exports.")

    def _stdout(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for ln in data.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev = json.loads(ln)
            except Exception:
                self.log.appendPlainText(ln)
                continue
            if not isinstance(ev, dict) or "event" not in ev:
                self.log.appendPlainText(ln)
                continue
            e = ev["event"]
            if e == "progress":
                self.bar.setValue(ev["done"])
                self.eta.setText(f"{ev['done']}/{ev['total']}  ETA {ev['eta_s'] / 60:.1f} min" +
                                 (f"  failed {ev['failed']}" if ev["failed"] else ""))
            elif e == "log":
                self.log.appendPlainText(ev["msg"])
            elif e == "error":
                self.log.appendPlainText(f"ERROR scene {ev['scene']}: {ev['msg']}")
            elif e == "start":
                self.log.appendPlainText(f"output: {ev['out']}  scenes: {ev['scenes']}  workers: {ev['workers']}")
            elif e == "finished":
                st = ev["stats"]
                self.log.appendPlainText(f"finished in {st.get('seconds')} s: images per split {st.get('splits')}, "
                                         f"instances per class {st.get('per_class')}")

    def _stderr(self):
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", "replace").rstrip()
        if data:
            self.log.appendPlainText(data)

    def _finished(self, code, _status):
        self.start.setEnabled(True)
        self.reexp.setEnabled(True)
        self.cancel.setEnabled(False)
        self.log.appendPlainText(f"[process exited with code {code} after {time.time() - self.t0:.1f} s]")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("syngt", "syngt")     # window geometry etc. only - never project data
        self.project = Project()
        self.project.subscribe(self._project_changed)
        self._autosave_timer = QTimer(self, singleShot=True, interval=1500)
        self._autosave_timer.timeout.connect(self._autosave)
        self._checks_timer = QTimer(self, singleShot=True, interval=300)
        self._rebuild_timer = QTimer(self, singleShot=True, interval=0)
        self._rebuild_timer.timeout.connect(self._rebuild_now)
        self._rebuild_all = False
        self._pending_tab = None            # tab to show once the pages exist (session restore)
        self.tabs = QTabWidget()
        self._cur_tab = None
        self.tabs.currentChanged.connect(self._tab_changed)
        self.preview = PreviewPanel(lambda: self.project.cfg)
        self.gen = GeneratePanel(self)
        self._checks_timer.timeout.connect(self.gen.refresh_checks)
        split = QSplitter(Qt.Horizontal)
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.tabs)
        left.addWidget(self.gen)
        left.setStretchFactor(0, 4)
        left.setStretchFactor(1, 1)
        split.addWidget(left)
        split.addWidget(self.preview)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        self.split, self.left = split, left
        self._sized = False
        self.setCentralWidget(split)
        self._menus()
        self.resize(1700, 1000)
        self._startup()

    def _startup(self):
        """Continue where the last session ended; else the last saved project; else the example."""
        try:
            if self.project.restore_session(self._session_file()):
                self._pending_tab = int(self.settings.value("session_tab", 0) or 0)
                legacy = self.settings.value("session_config_path", "") or ""   # first session format
                if self.project.path is None and legacy and Path(legacy).exists():
                    self.project.path = Path(legacy)
                    self.settings.remove("session_config_path")
                self.statusBar().showMessage("continuing with the settings of the last session", 6000)
                return
        except Exception as e:
            self.statusBar().showMessage(f"could not restore the last session ({e})", 8000)
        last = self.settings.value("last_config", "")
        for cand in (last, EXAMPLE):
            if cand and Path(cand).exists() and self.load(Path(cand)):
                return
        self.project.new()

    # ------------------------------------------------------------------ menus
    def _menus(self):
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for text, key, fn in [("&New project", QKeySequence.New, self.new), ("&Open config…", QKeySequence.Open, self.open),
                              ("&Save", QKeySequence.Save, self.save), ("Save &as…", QKeySequence.SaveAs, self.save_as),
                              ("Load example project", None, lambda: self._confirm_discard() and self.load(EXAMPLE))]:
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn)
            f.addAction(a)
        f.addSeparator()
        a = QAction("&Quit", self)
        a.triggered.connect(self.close)
        f.addAction(a)
        e = mb.addMenu("&Edit")
        self.a_undo = QAction("&Undo", self)
        self.a_undo.setShortcut(QKeySequence.Undo)
        self.a_undo.triggered.connect(self.undo)
        e.addAction(self.a_undo)
        self.a_redo = QAction("&Redo", self)
        self.a_redo.setShortcut(QKeySequence.Redo)
        self.a_redo.triggered.connect(self.redo)
        e.addAction(self.a_redo)
        t = mb.addMenu("&Tools")
        for text, key, fn in [
                ("Render preview", "F5", self.preview.render),
                ("New preview scene", "F6", self.preview.new_scene),
                ("Phone photo → ground texture…", None, lambda: self._rectifier(None)),
                ("Build clean background plates from real photos…", None, self._plates),
                ("Create camera look profile from real photos…", None, self._create_look),
                ("Match camera look to real photos (one-off)…", None, self._match_look),
                ("Dataset browser…", None, lambda: DatasetBrowser(self, str(self.project.output_dir())).exec()),
                ("Clear mesh cache", None, self._clear_cache)]:
            a = QAction(text, self)
            if key:
                a.setShortcut(key)
            a.triggered.connect(fn)
            t.addAction(a)
        t.addSeparator()
        if os.name == "nt":   # Optimus laptops only; Linux/Jetson picks the GPU by itself
            a = QAction("Use high-performance (NVIDIA) GPU for Python", self)
            a.triggered.connect(lambda: self._gpu_pref(True))
            t.addAction(a)
            a = QAction("Reset GPU preference to Windows default", self)
            a.triggered.connect(lambda: self._gpu_pref(False))
            t.addAction(a)
        a = QAction("Show OpenGL renderer (GPU)", self)
        a.triggered.connect(lambda: QMessageBox.information(self, "OpenGL renderer", "Preview renderer: " + self._gl()))
        t.addAction(a)
        h = mb.addMenu("&Help")
        a = QAction("Conventions && formats", self)
        a.triggered.connect(self._help)
        h.addAction(a)

    def _gl(self) -> str:
        return str(getattr(self.preview.worker.renderer, "info", {}).get("GL_RENDERER", "not started yet"))

    def _create_look(self):
        from .lookdialog import CreateLookDialog
        CreateLookDialog(self, self.project).exec()
        self._schedule_rebuild(False)       # the list of look profiles changed

    def _plates(self):
        from .realism import build_plates_dialog
        build_plates_dialog(self, self.project)

    def _match_look(self):
        from .realism import match_look_dialog
        match_look_dialog(self, self.project)

    def _gpu_pref(self, high: bool):
        from ..gpu import set_gpu_preference
        try:
            paths = set_gpu_preference(high)
        except Exception as e:
            QMessageBox.warning(self, "GPU preference", str(e))
            return
        QMessageBox.information(self, "GPU preference", (
            ("Marked as 'High performance' (per user):\n" if high else "Preference removed for:\n") +
            "\n".join(paths) + f"\n\nCurrent preview renderer: {self._gl()}\n"
            "Restart the application for the change to take effect (generation runs in new processes "
            "and picks it up immediately)."))

    def _clear_cache(self):
        from ..assets import clear_cache
        clear_cache()
        self.statusBar().showMessage("mesh cache cleared", 4000)

    def _help(self):
        QMessageBox.information(self, "Conventions", (
            "World: z up, metres, support plane z=0.\nCamera: OpenCV (x right, y down, z forward).\n"
            "T_a_b maps points from frame b to frame a. Poses: T_cam_obj (native), cam_R_m2c / cam_t_m2c [mm] (BOP).\n"
            "Object frame = CAD frame (unless 're-centre' is enabled).\n\n"
            "Native: rgb/, depth/ (uint16, depth_scale_mm), masks/ (uint16 instance ids), annotations/<split>/<stem>.json, "
            "frames/<split>/<scene>_<view>.json (multi-view groups), cameras.json (rig).\n"
            "BOP: bop/<split>_pbr/<scene>/ with rgb, depth, mask, mask_visib, scene_gt(_info), scene_camera "
            "(+ _<cam> suffixes for multi-camera, BOP-H3 style), bop/models (PLY, mm, models_info with symmetries).\n"
            "COCO: coco/<split>.json (+ obb, keypoints, pose, amodal bbox).\n"
            "YOLO: yolo*/images|labels/<split>, data.yaml (detect, obb, seg, pose).\n\n"
            "CLI: python -m syngt generate cfg.yaml | export <dataset> | preview cfg.yaml | presets"))

    # ------------------------------------------------------------------ pages
    def _page_factories(self):
        p = self.project
        return [("★ Quick setup", lambda: QuickSetupPage(p, self.report_changes, lambda: self._schedule_rebuild(False))),
                ("Objects", lambda: ObjectsPage(p)),
                ("Placement", lambda: PlacementPage(p)),
                ("Scene && Light", lambda: ScenePage(p, self._rectifier)),
                ("Cameras", lambda: CamerasPage(p)),
                ("Viewpoints", lambda: ViewpointsPage(p)),
                ("Sensor", lambda: SensorPage(p)),
                ("Output", lambda: OutputPage(p))]

    def show_tab(self, which) -> None:
        """Select a tab by index or by (part of) its title."""
        if isinstance(which, int):
            if 0 <= which < self.tabs.count():
                self.tabs.setCurrentIndex(which)
            return
        for i in range(self.tabs.count()):
            if which and which.replace("&", "") in self.tabs.tabText(i).replace("&&", "&").replace("&", ""):
                self.tabs.setCurrentIndex(i)
                return

    def _make_page(self, i: int):
        w = self._page_factories()[i][1]()
        w._built_revision = self.project.revision
        return w

    @staticmethod
    def _page_state(w) -> dict:
        """What to keep when a page is rebuilt: scroll positions and the selected list row."""
        return {"scroll": [sa.verticalScrollBar().value() for sa in w.findChildren(QScrollArea)],
                "row": w.list.currentRow() if hasattr(w, "list") else None}

    @staticmethod
    def _restore_page_state(w, st: dict) -> None:
        row = st.get("row")
        if row is not None and row >= 0 and hasattr(w, "list") and row < w.list.count():
            w.list.setCurrentRow(row)
        areas = w.findChildren(QScrollArea)

        def scroll():
            for sa, val in zip(areas, st.get("scroll", [])):
                try:
                    sa.verticalScrollBar().setValue(val)
                except RuntimeError:    # page replaced again meanwhile
                    pass
        QTimer.singleShot(0, scroll)    # after the layout pass, when the scroll range is known

    def _replace_page(self, i: int) -> None:
        old = self.tabs.widget(i)
        state = self._page_state(old) if old is not None else {}
        title = self.tabs.tabText(i)
        new = self._make_page(i)
        was_current = self.tabs.currentIndex() == i
        self.tabs.blockSignals(True)
        self.tabs.removeTab(i)
        self.tabs.insertTab(i, new, title)
        if was_current:
            self.tabs.setCurrentIndex(i)
        self.tabs.blockSignals(False)
        if old is not None:
            old.deleteLater()
        self._restore_page_state(new, state)

    def _build_pages(self):
        cur = max(0, self.tabs.currentIndex())
        self.tabs.blockSignals(True)
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        for i, (title, _) in enumerate(self._page_factories()):
            self.tabs.addTab(self._make_page(i), title)
        self.tabs.setCurrentIndex(cur)
        self.tabs.blockSignals(False)
        self._cur_tab = self.tabs.currentIndex()

    def _schedule_rebuild(self, everything: bool) -> None:
        self._rebuild_all = self._rebuild_all or everything
        self._rebuild_timer.start()

    def _rebuild_now(self):
        if QApplication.activeModalWidget() is not None:    # a dialog opened from a page is still running:
            self._rebuild_timer.start(150)                  # never delete its parent underneath it
            return
        if self._rebuild_all or self.tabs.count() == 0:
            self._rebuild_all = False
            self._build_pages()
            if self._pending_tab is not None:
                tab, self._pending_tab = self._pending_tab, None
                self.show_tab(tab)
        elif self.tabs.currentIndex() >= 0:
            self._replace_page(self.tabs.currentIndex())

    def _tab_changed(self, i: int):
        """A tab that is older than the last change is rebuilt when it is shown."""
        prev = self.tabs.widget(self._cur_tab) if self._cur_tab is not None else None
        if prev is not None and not self._rebuild_timer.isActive():
            prev._built_revision = self.project.revision    # it was visible, so it shows the current values
        self._cur_tab = i
        w = self.tabs.widget(i)
        if w is not None and getattr(w, "_built_revision", -1) != self.project.revision:
            self._replace_page(i)

    def _rectifier(self, path_edit):
        RectifyDialog(self, path_edit).exec()

    # ------------------------------------------------------------------ project events
    def _project_changed(self, kind: str):
        """EDIT: one bound field changed (its widget already shows it). BULK: many settings changed, the visible
        page must re-read them. RESET: the whole configuration object was replaced."""
        if kind == RESET:
            self._schedule_rebuild(True)
        elif kind == BULK:
            self._schedule_rebuild(False)
        elif kind == EDIT:
            w = self.tabs.currentWidget()
            if w is not None:
                w._built_revision = self.project.revision
                if hasattr(w, "on_project_changed"):
                    w.on_project_changed(kind)
        self._title()
        self.a_undo.setEnabled(self.project.can_undo())
        self.a_redo.setEnabled(self.project.can_redo())
        self._autosave_timer.start()
        self._checks_timer.start()
        self.preview.settings_changed()

    def report_changes(self, label: str, changed: list) -> None:
        """Tell the user what a bulk action touched - nothing changes behind their back."""
        if label and changed:
            self.statusBar().showMessage(f"'{label}' changed {len(changed)} setting(s): {summarise_changes(changed)}"
                                         f"   –   Ctrl+Z to undo", 12000)
        elif label:
            self.statusBar().showMessage(f"'{label}': nothing to change, the settings already match", 6000)

    def undo(self):
        label = self.project.undo()
        self.statusBar().showMessage(f"undone: {label}" if label else "nothing to undo", 5000)

    def redo(self):
        label = self.project.redo()
        self.statusBar().showMessage(f"redone: {label}" if label else "nothing to redo", 5000)

    # ------------------------------------------------------------------ session (auto save / restore)
    @staticmethod
    def _session_file() -> Path:
        # not next to QSettings: on Windows those live in the registry, which has no file location
        return user_dir() / "syngt_session.yaml"

    def _autosave(self):
        """Store the current project (incl. unsaved edits) so the next start continues exactly here."""
        try:
            self.project.save_session(self._session_file())
        except Exception as e:  # never block the GUI on this
            self.statusBar().showMessage(f"auto save failed: {e}", 5000)

    def _title(self):
        name = str(self.project.path) if self.project.path else "untitled"
        self.setWindowTitle(f"syngt - synthetic GT generator - {name}{' *' if self.project.dirty else ''}")

    # ------------------------------------------------------------------ files
    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def new(self):
        if self._confirm_discard():
            self.project.new()

    def open(self):
        if not self._confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open config", str(self.project.path or self.project.root),
                                           "Config (*.yaml *.yml *.json)")
        if p:
            self.load(Path(p))

    def load(self, p: Path) -> bool:
        try:
            self.project.load(p)
        except Exception as e:
            QMessageBox.warning(self, "Open failed", f"{p}\n\n{e}")
            return False
        self.settings.setValue("last_config", str(self.project.path))
        return True

    def save(self) -> bool:
        if self.project.path is None:
            return self.save_as()
        try:
            self.project.save()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return False
        self._title()
        self.statusBar().showMessage(f"saved {self.project.path}", 3000)
        self._autosave_timer.start()
        return True

    def save_as(self) -> bool:
        p, _ = QFileDialog.getSaveFileName(self, "Save config", str(self.project.path or (self.project.root / "project.yaml")),
                                           "YAML (*.yaml);;JSON (*.json)")
        if not p:
            return False
        if not Path(p).suffix:
            p += ".yaml"
        self.project.path = Path(p).resolve()
        self.settings.setValue("last_config", str(self.project.path))
        return self.save()

    # ------------------------------------------------------------------ window
    def showEvent(self, e):
        super().showEvent(e)
        if not self._sized and self.settings.value("session_geometry") is not None:
            self._sized = True
            self.restoreGeometry(self.settings.value("session_geometry"))
            if self.settings.value("session_split") is not None:
                self.split.restoreState(self.settings.value("session_split"))
            if self.settings.value("session_split_left") is not None:
                self.left.restoreState(self.settings.value("session_split_left"))
            else:
                self.left.setSizes([int(self.height() * 0.72), int(self.height() * 0.28)])
        if not self._sized:  # settings ~42 %, preview ~58 % of the window; settings above a compact log
            self._sized = True
            w = max(self.width(), 800)
            self.split.setSizes([int(w * 0.42), int(w * 0.58)])
            self.left.setSizes([int(self.height() * 0.72), int(self.height() * 0.28)])

    def closeEvent(self, e):
        # no "save changes?" prompt: the session (incl. unsaved edits) is stored and restored on the next start
        if self.gen.proc is not None and self.gen.proc.state() != QProcess.NotRunning:
            if QMessageBox.question(self, "Quit", "Generation is running. Kill it?") != QMessageBox.Yes:
                e.ignore()
                return
            self.gen.proc.kill()
        self._autosave()
        self.settings.setValue("session_tab", self.tabs.currentIndex())
        self.settings.setValue("session_geometry", self.saveGeometry())
        self.settings.setValue("session_split", self.split.saveState())
        self.settings.setValue("session_split_left", self.left.saveState())
        self.settings.sync()
        self.preview.shutdown()
        e.accept()
