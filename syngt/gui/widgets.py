"""Widgets bound to the fields of a project's configuration.

All edits go through a :class:`Binder`, which hands them to ``Project.set`` / ``Project.touch`` - so the dirty flag,
undo history, auto-save, preview and pre-flight checks see every change, and unchanged values notify nobody.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QScrollBar, QSpinBox, QVBoxLayout, QWidget)

from ..project import Project


class WheelGuard(QObject):
    """Application-wide event filter: the mouse wheel only ever scrolls pages.

    By default Qt lets the wheel change any combo box, spin box or slider under the cursor - scrolling through a
    settings page would silently change whatever the pointer passes over (another profile, another count ...).
    """

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and isinstance(obj, (QComboBox, QAbstractSpinBox, QAbstractSlider)) \
                and not isinstance(obj, QScrollBar):
            area = obj.parentWidget()
            while area is not None and not isinstance(area, QAbstractScrollArea):
                area = area.parentWidget()
            if area is not None:
                QApplication.sendEvent(area.viewport(), ev)     # scroll the page instead
            ev.accept()
            return True                                         # ... and never the editor
        return False


def muted_style(extra: str = "") -> str:
    """Style for secondary text (hints, column headers). Derived from the real text and window colours, so it
    is readable on every desktop theme (the palette's 'mid' role is nearly invisible on some)."""
    pal = QApplication.palette()
    t, w = pal.color(pal.ColorRole.WindowText), pal.color(pal.ColorRole.Window)
    r, g, b = [int(round(0.62 * a + 0.38 * c)) for a, c in ((t.red(), w.red()), (t.green(), w.green()),
                                                             (t.blue(), w.blue()))]
    return f"color: rgb({r},{g},{b});" + (" " + extra if extra else "")


def _tip(w: QWidget, tip: str) -> QWidget:
    if tip:
        w.setToolTip(tip)
    return w


class Binder:
    """Factory for editors of config fields. ``obj`` is a config (sub-)object of ``project.cfg``."""

    def __init__(self, project: Project):
        self.project = project

    def _commit(self, obj, attr: str, value, on_change=None) -> None:
        if self.project.set(obj, attr, value) and on_change:
            on_change(value)

    def touch(self, label: str = "edit") -> None:
        """After changing the config in place (lists, several fields)."""
        self.project.touch(label)

    # ------------------------------------------------------------------ scalars
    def dspin(self, obj, attr: str, lo=-1e9, hi=1e9, step=0.1, dec=3, suffix="", tip="", on_change=None) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(" " + suffix)
        w.setValue(float(getattr(obj, attr)))
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self._commit(obj, attr, float(v), on_change))
        w.setMinimumWidth(80)
        return _tip(w, tip)

    def ispin(self, obj, attr: str, lo=-10**9, hi=10**9, step=1, suffix="", tip="", on_change=None,
              special: str = "") -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(" " + suffix)
        if special:
            w.setSpecialValueText(special)   # shown at the minimum (e.g. 0 = "auto")
        w.setValue(int(getattr(obj, attr)))
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self._commit(obj, attr, int(v), on_change))
        return _tip(w, tip)

    def check(self, obj, attr: str, text="", tip="", on_change=None) -> QCheckBox:
        w = QCheckBox(text)
        w.setChecked(bool(getattr(obj, attr)))
        w.toggled.connect(lambda v: self._commit(obj, attr, bool(v), on_change))
        return _tip(w, tip)

    def combo(self, obj, attr: str, options: Sequence, tip="", on_change=None) -> QComboBox:
        """options: list of values or (value, label)."""
        w = QComboBox()
        opts = [(o, o) if not isinstance(o, tuple) else o for o in options]
        for v, lab in opts:
            w.addItem(str(lab), v)
        cur = getattr(obj, attr)
        w.setCurrentIndex(next((i for i, (v, _) in enumerate(opts) if v == cur), 0))
        w.currentIndexChanged.connect(lambda i: self._commit(obj, attr, w.itemData(i), on_change))
        return _tip(w, tip)

    def line(self, obj, attr: str, tip="", on_change=None) -> QLineEdit:
        w = QLineEdit(str(getattr(obj, attr) or ""))
        w.editingFinished.connect(lambda: self._commit(obj, attr, w.text(), on_change))
        return _tip(w, tip)

    # ------------------------------------------------------------------ ranges / vectors / sets
    def range_w(self, r, lo=-1e9, hi=1e9, step=0.1, dec=3, suffix="", tip="", dist=True) -> QWidget:
        """Editor for a config.Range (min, max, distribution)."""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("min"))
        h.addWidget(self.dspin(r, "min", lo, hi, step, dec, suffix), 1)
        h.addWidget(QLabel("max"))
        h.addWidget(self.dspin(r, "max", lo, hi, step, dec, suffix), 1)
        if dist:
            c = self.combo(r, "dist", [("uniform", "uniform"), ("loguniform", "log-uniform"),
                                       ("normal", "normal (clipped)")])
            c.setMaximumWidth(120)
            h.addWidget(c)
        return _tip(box, tip)

    def irange_w(self, r, lo=0, hi=10**6, tip="") -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("min"))
        h.addWidget(self.ispin(r, "min", lo, hi), 1)
        h.addWidget(QLabel("max"))
        h.addWidget(self.ispin(r, "max", lo, hi), 1)
        return _tip(box, tip)

    def vec_w(self, obj, attr: str, n: int, lo=-1e9, hi=1e9, step=0.01, dec=4, labels: Optional[Sequence[str]] = None,
              suffix="", tip="", start: int = 0) -> QWidget:
        """Editors for elements start..start+n-1 of a list field (the list is padded with zeros if shorter)."""
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        vals = list(getattr(obj, attr))
        for k in range(n):
            i = start + k
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(step)
            w.setKeyboardTracking(False)
            if suffix:
                w.setSuffix(" " + suffix)
            w.setValue(float(vals[i]) if i < len(vals) else 0.0)

            def f(v, i=i):
                arr = list(getattr(obj, attr))
                arr += [0.0] * (i + 1 - len(arr))
                arr[i] = float(v)
                self._commit(obj, attr, arr)
            w.valueChanged.connect(f)
            if labels:
                h.addWidget(QLabel(labels[k]))
            h.addWidget(w, 1)
        return _tip(box, tip)

    def flags(self, obj, attr: str, options: Sequence, columns: int = 0, locked: Sequence[str] = ()) -> QWidget:
        """Check boxes for a list-of-names field. options: values or (value, label); locked: always on."""
        from PySide6.QtWidgets import QGridLayout
        box = QWidget()
        g = QGridLayout(box)
        g.setContentsMargins(0, 0, 0, 0)
        opts = [(o, o) if not isinstance(o, tuple) else o for o in options]
        for k, (val, lab) in enumerate(opts):
            cb = QCheckBox(str(lab))
            cb.setChecked(val in getattr(obj, attr) or val in locked)
            cb.setEnabled(val not in locked)

            def f(on, val=val):
                cur = [x for x in getattr(obj, attr) if x != val]
                if on:   # keep the canonical order of the options
                    cur = [v for v, _ in opts if v in cur or v == val] + [x for x in cur if x not in dict(opts)]
                self._commit(obj, attr, cur)
            cb.toggled.connect(f)
            if columns:
                g.addWidget(cb, k // columns, k % columns)
            else:
                g.addWidget(cb, 0, k)
        return box

    def path(self, obj, attr: str, mode: str = "dir", filt: str = "", tip: str = "") -> "PathEdit":
        return PathEdit(self, obj, attr, mode, filt, tip)

    def color(self, obj, attr: str, tip: str = "", on_pick=None) -> "ColorButton":
        return ColorButton(self, obj, attr, tip, on_pick)


class PathEdit(QWidget):
    changed = Signal(str)

    def __init__(self, binder: Binder, obj, attr: str, mode: str = "dir", filt: str = "", tip: str = ""):
        super().__init__()
        self.binder, self.obj, self.attr, self.mode, self.filt = binder, obj, attr, mode, filt
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit(str(getattr(obj, attr) or ""))
        self.edit.editingFinished.connect(self._set)
        b = QPushButton("…")
        b.setFixedWidth(28)
        b.clicked.connect(self._browse)
        h.addWidget(self.edit, 1)
        h.addWidget(b)
        _tip(self, tip)

    def _set(self):
        self.binder._commit(self.obj, self.attr, self.edit.text(), self.changed.emit)

    def set_path(self, p: str):
        self.edit.setText(p)
        self._set()

    def _browse(self):
        cur = self.edit.text()
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Select folder", cur)
        elif self.mode == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Select file", cur, self.filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select file", cur, self.filt)
        if p:
            self.set_path(p)


class ColorButton(QPushButton):
    def __init__(self, binder: Binder, obj, attr: str, tip="", on_pick: Optional[Callable[[], None]] = None):
        super().__init__()
        self.binder, self.obj, self.attr, self.on_pick = binder, obj, attr, on_pick
        self.setFixedWidth(90)
        self._refresh()
        self.clicked.connect(self._pick)
        _tip(self, tip)

    def _refresh(self):
        c = getattr(self.obj, self.attr)
        r, g, b = [int(round(max(0, min(1, x)) * 255)) for x in c[:3]]
        self.setStyleSheet(f"background-color: rgb({r},{g},{b}); color: {'black' if (r + g + b) > 380 else 'white'};")
        self.setText(f"#{r:02x}{g:02x}{b:02x}")

    def _pick(self):
        c = getattr(self.obj, self.attr)
        q = QColorDialog.getColor(QColor.fromRgbF(*[max(0, min(1, x)) for x in c[:3]]), self, "Colour")
        if q.isValid():
            with self.binder.project.batch("colour"):
                setattr(self.obj, self.attr, [round(q.redF(), 4), round(q.greenF(), 4), round(q.blueF(), 4)])
                if self.on_pick:
                    self.on_pick()      # e.g. switch the colour mode so that the picked colour is used
            self._refresh()


def group(title: str, rows: Sequence[Tuple[str, QWidget]], tip: str = "") -> QGroupBox:
    g = QGroupBox(title)
    f = QFormLayout(g)
    f.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    for lab, w in rows:
        if lab is None:
            f.addRow(w)
        else:
            f.addRow(lab, w)
    if tip:
        g.setToolTip(tip)
    return g


def scroll_page(*widgets: QWidget, stretch=True) -> QScrollArea:
    inner = QWidget()
    v = QVBoxLayout(inner)
    for w in widgets:
        v.addWidget(w)
    if stretch:
        v.addStretch(1)
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    return sa


def note(text: str) -> QLabel:
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet(muted_style("font-size: 11px;"))
    l.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return l


def hbox(*ws, stretch_last=False) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    for x in ws:
        if isinstance(x, int):
            h.addStretch(x)
        else:
            h.addWidget(x)
    if stretch_last:
        h.addStretch(1)
    return w
