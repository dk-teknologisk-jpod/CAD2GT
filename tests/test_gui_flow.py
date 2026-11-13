"""The window as the user drives it (offscreen): no setting may change, get lost or be ignored behind their back.

Run:  QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_flow.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (QAbstractSlider, QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QScrollBar, QSpinBox)

from syngt.config import ObjectModel  # noqa: E402
from syngt.gui.widgets import ColorButton, WheelGuard  # noqa: E402

BRICK = str(ROOT / "examples" / "models" / "brick_50x10x5.stl")


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    guard = WheelGuard(a)
    a.installEventFilter(guard)
    yield a


@pytest.fixture()
def win(app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))    # never touch the real user settings
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "cfg"))
    from syngt.gui import main_window as mw
    monkeypatch.setattr(mw.PreviewPanel, "render", lambda self: None)   # no rendering in these tests
    monkeypatch.setattr(mw, "EXAMPLE", tmp_path / "none.yaml")
    w = mw.MainWindow()
    w.project.cfg.objects.models.append(ObjectModel(path=BRICK, name="brick", unit="mm"))
    w.project.touch("add part")
    w.show()
    pump(app)
    yield w
    w._autosave_timer.stop()
    w.preview.shutdown()
    w.deleteLater()
    pump(app)


def pump(app, n=6):
    for _ in range(n):
        app.processEvents()


def wheel(app, widget, steps=3):
    for _ in range(steps):
        c = widget.rect().center()
        ev = QWheelEvent(QPointF(c), QPointF(widget.mapToGlobal(c)), QPoint(0, 0), QPoint(0, -120), Qt.NoButton,
                         Qt.NoModifier, Qt.ScrollUpdate, False)
        app.sendEvent(widget, ev)
    pump(app)


def test_mouse_wheel_never_changes_a_setting(app, win):
    """Scrolling a settings page must not alter combos / spin boxes / sliders the pointer passes over."""
    for tab in range(win.tabs.count()):
        win.show_tab(tab)
        pump(app)
        before = win.project.cfg.to_dict()
        page = win.tabs.currentWidget()
        widgets = [w for cls in (QComboBox, QSpinBox, QDoubleSpinBox, QAbstractSlider) for w in page.findChildren(cls)
                   if not isinstance(w, QScrollBar) and w.isVisible()]
        assert widgets, win.tabs.tabText(tab)
        for w in widgets:
            w.setFocus()            # even a focused editor must not react
            wheel(app, w)
        assert win.project.cfg.to_dict() == before, f"wheel changed settings on tab {win.tabs.tabText(tab)}"


def test_picking_a_colour_makes_it_the_colour(app, win, monkeypatch):
    mat = win.project.cfg.objects.models[0].material
    mat.mode = "random_color"
    win.project.touch("mode")
    win.show_tab("Objects")
    pump(app)
    btn = next(b for b in win.tabs.currentWidget().findChildren(ColorButton) if b.obj is mat)
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor(0, 255, 0)))
    btn._pick()
    pump(app)
    assert mat.mode == "fixed_color" and mat.color == [0.0, 1.0, 0.0]
    assert win.project.dirty


def test_profile_reports_changes_and_undo_restores(app, win):
    win.show_tab(0)
    pump(app)
    before = win.project.cfg.to_dict()
    quick = win.tabs.currentWidget()
    combo = next(c for c in quick.findChildren(QComboBox) if c.findData("pile") >= 0)
    combo.setCurrentIndex(combo.findData("pile"))
    combo.activated.emit(combo.currentIndex())
    pump(app)
    assert win.project.cfg.placement.sequential_drop is True
    assert "changed" in win.statusBar().currentMessage() and "placement" in win.statusBar().currentMessage()
    assert win.tabs.currentWidget() is not quick                   # page re-read the config
    win.undo()
    pump(app)
    assert win.project.cfg.to_dict() == before


def test_bulk_change_never_leaves_widgets_bound_to_dead_objects(app, win):
    """A look profile replaces cfg.sensor; an edit on the visible Sensor tab afterwards must reach the new one."""
    win.show_tab("Sensor")
    pump(app)
    looks = sorted((ROOT / "looks").glob("*.json"))
    if not looks:
        pytest.skip("no look profile in looks/")
    old_sensor = win.project.cfg.sensor
    win.project.apply_look(str(looks[0]))
    pump(app)
    assert win.project.cfg.sensor is not old_sensor
    box = next(c for c in win.tabs.currentWidget().findChildren(QCheckBox) if c.text() == "Enable sensor effects")
    box.setChecked(not win.project.cfg.sensor.enabled)
    pump(app)
    assert win.project.cfg.sensor.enabled == box.isChecked()


def test_every_toggle_reaches_the_project(app, win):
    """Format and distractor-shape check boxes used to change the config without telling anyone."""
    win.show_tab("Output")
    pump(app)
    win.project.dirty = False
    box = next(c for c in win.tabs.currentWidget().findChildren(QCheckBox) if c.text().startswith("yolo_seg"))
    box.setChecked(True)
    assert "yolo_seg" in win.project.cfg.output.formats and win.project.dirty
    win.show_tab("Objects")
    pump(app)
    win.project.dirty = False
    box = next(c for c in win.tabs.currentWidget().findChildren(QCheckBox) if c.text() == "torus")
    box.setChecked(True)
    assert "torus" in win.project.cfg.objects.distractors.shapes and win.project.dirty


def test_other_tabs_show_edits_made_elsewhere(app, win):
    win.show_tab("Output")
    pump(app)
    win.show_tab(0)
    pump(app)
    win.project.set(win.project.cfg.generation, "num_scenes", 123)
    win.show_tab("Output")
    pump(app)
    spins = [s.value() for s in win.tabs.currentWidget().findChildren(QSpinBox)]
    assert 123 in spins


def test_session_continues_where_it_stopped(app, win):
    win.project.set(win.project.cfg.generation, "num_scenes", 77)
    win._autosave()
    from syngt.project import Project
    q = Project()
    assert q.restore_session(win._session_file()) and q.cfg.generation.num_scenes == 77


def test_checks_panel_lists_problems(app, win):
    win.project.set(win.project.cfg.scene.background, "mode", "image_folder")
    win.gen.refresh_checks()
    texts = [win.gen.checks.item(i).text() for i in range(win.gen.checks.count())]
    assert any("photo folder" in t for t in texts), texts
    win.project.cfg.objects.models.clear()
    win.project.touch("remove")
    win.gen.refresh_checks()
    assert "fix" in win.gen.ready.text()


def test_camera_page_keeps_names_unique_and_user_settings(app, win):
    from syngt.gui.pages import CamerasPage
    win.show_tab("Cameras")
    pump(app)
    page = win.tabs.currentWidget()
    assert isinstance(page, CamerasPage)
    cams = win.project.cfg.cameras
    cams[0].intrinsics_jitter, cams[0].depth.enabled = 0.02, False
    page.list.setCurrentRow(0)
    page._dup()
    page.list.setCurrentRow(0)
    page._dup()
    names = [c.name for c in win.project.cfg.cameras]
    assert len(set(names)) == 3, names                               # duplicates used to share one file name
    page.list.setCurrentRow(0)
    page._apply_preset()
    c = win.project.cfg.cameras[0]
    assert c.intrinsics_jitter == 0.02 and c.depth.enabled is False   # a preset changes the optics only
    assert not any("same file name" in str(i) for i in win.project.issues())


def test_class_name_follows_the_cad_file(app, win):
    from syngt.gui.widgets import PathEdit
    m = win.project.cfg.objects.models[0]
    m.name = m.cad_name
    win.project.touch("name")
    win.show_tab("Objects")
    pump(app)
    other = str(ROOT / "examples" / "models" / "gear.stl")
    edit = next(e for e in win.tabs.currentWidget().findChildren(PathEdit) if e.obj is m)
    edit.set_path(other)
    pump(app)
    assert m.class_name == "gear"                       # was the old file's name: now the new file's
    m.name = "my part"
    edit.set_path(BRICK)
    assert m.class_name == "my part"                    # a name the user typed is kept
    win.tabs.currentWidget()._cad_names()
    pump(app)
    assert m.class_name == "brick_50x10x5"
