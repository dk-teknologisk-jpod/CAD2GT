"""Create a camera look profile from real photos (runs the closed-loop tuning in a worker thread)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout)



class _Worker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    cancelled = False       # set by the dialog when it is closed; checked between the tuning steps

    def _check(self):
        if self.cancelled:
            raise InterruptedError("cancelled")

    def _log(self, msg: str):
        self._check()
        self.log.emit(msg)

    def _progress(self, i: int, n: int):
        self._check()
        self.progress.emit(i, n)

    @Slot(object)
    def run(self, job):
        try:
            from ..lookprofile import create_profile
            prof = create_profile(job["name"], job["folders"], job["cfg"], log=self._log, progress=self._progress)
            self._check()
            self.done.emit(prof)
        except InterruptedError:
            pass
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class CreateLookDialog(QDialog):
    start = Signal(object)

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Create camera look profile from real photos")
        self.resize(820, 600)
        self.project = project
        self.result_path = None
        self._closed = False
        self.name = QLineEdit("my camera")
        self.list = QListWidget()
        b_add = QPushButton("Add folder…")
        b_add.clicked.connect(self._add)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        self.make_default = QCheckBox("Use as default for new projects")
        self.make_default.setChecked(True)
        self.apply_now = QCheckBox("Apply to the current project")
        self.apply_now.setChecked(True)
        self.bar = QProgressBar()
        self.logw = QPlainTextEdit()
        self.logw.setReadOnly(True)
        info = QLabel("Pick folders with <b>real images from the camera</b> you want to imitate (labelled images are "
                      "fine – annotated objects are masked when measuring the table colours). The tool measures "
                      "brightness, contrast, colour saturation, colour cast, noise and sharpness, then renders "
                      "synthetic frames of the <b>current project</b> and tunes the sensor settings until they match "
                      "(about 30-60 s). The result is saved in <i>looks/</i> and can be applied to any project.")
        info.setWordWrap(True)
        self.bb = QDialogButtonBox()
        self.go = self.bb.addButton("Create profile", QDialogButtonBox.AcceptRole)
        self.bb.addButton(QDialogButtonBox.Close)
        self.bb.rejected.connect(self.reject)
        self.go.clicked.connect(self._run)
        v = QVBoxLayout(self)
        v.addWidget(info)
        h = QHBoxLayout()
        h.addWidget(QLabel("Profile name:"))
        h.addWidget(self.name, 1)
        v.addLayout(h)
        v.addWidget(QLabel("Real image folders:"))
        v.addWidget(self.list)
        h2 = QHBoxLayout()
        h2.addWidget(b_add)
        h2.addWidget(b_rm)
        h2.addStretch(1)
        h2.addWidget(self.make_default)
        h2.addWidget(self.apply_now)
        v.addLayout(h2)
        v.addWidget(self.bar)
        v.addWidget(self.logw, 1)
        v.addWidget(self.bb)
        self.thread = QThread(self)
        self.worker = _Worker()
        self.worker.moveToThread(self.thread)
        self.start.connect(self.worker.run)
        self.worker.log.connect(self.logw.appendPlainText)
        # bound slots, never lambdas: a lambda would run in the worker thread and touch widgets from there
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.thread.start()

    @Slot(int, int)
    def _progress(self, i: int, n: int):
        self.bar.setRange(0, n)
        self.bar.setValue(i)

    @Slot(str)
    def _failed(self, m: str):
        self.logw.appendPlainText("FAILED: " + m)
        self.go.setEnabled(True)

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with real camera images")
        if d:
            self.list.addItem(d)

    def _run(self):
        folders = [self.list.item(i).text() for i in range(self.list.count())]
        if not folders:
            self.logw.appendPlainText("Add at least one folder.")
            return
        cfg = self.project.cfg.copy_deep()
        if not cfg.enabled_models():  # need something to render: use the example brick
            from ..config import ObjectModel
            cfg.objects.models = [ObjectModel(path=str(Path(__file__).resolve().parents[2] / "examples" / "models" /
                                                       "brick_50x10x5.stl"), name="brick", unit="mm")]
        from ..lookprofile import LOOKS_DIR, _slug
        name = self.name.text().strip() or "camera"
        target = LOOKS_DIR / (_slug(name) + ".json")
        if target.exists() and QMessageBox.question(
                self, "Profile exists", f"A profile named '{name}' already exists ({target.name}).\n"
                                        f"Replace it?") != QMessageBox.Yes:
            return
        self.go.setEnabled(False)
        self.logw.appendPlainText(f"measuring {len(folders)} folder(s) and tuning …")
        self.start.emit({"name": self.name.text().strip() or "camera", "folders": folders, "cfg": cfg})

    @Slot(object)
    def _done(self, prof):
        from ..lookprofile import report, save_profile, set_default_profile
        if self._closed:        # the dialog was closed while tuning: closing means "cancel", nothing is saved
            return
        path = save_profile(prof)
        self.result_path = path
        self.logw.appendPlainText("\n" + report(prof) + f"\n\nsaved {path}")
        if self.make_default.isChecked():
            set_default_profile(path)
            self.logw.appendPlainText("set as default for new projects")
        if self.apply_now.isChecked():
            changed = self.project.apply_look(path)
            self.logw.appendPlainText(f"applied to the current project ({len(changed)} settings changed)")
        self.go.setEnabled(True)

    def done(self, r):
        self._closed = True             # a result that arrives after this is discarded (see _done)
        self.worker.cancelled = True    # the worker stops at its next step (a few seconds at most)
        self.thread.quit()
        self.hide()
        self.thread.wait(60000)
        super().done(r)
