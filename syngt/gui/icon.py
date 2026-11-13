"""The application icon: a CAD part (cube) inside the corner brackets of a bounding box - "labelled parts".

It is drawn in code, so it is sharp at every size and the app needs no image file to show it. ``write_icon_files``
exports PNG / ICO copies for desktop shortcuts (syngt/gui/assets/).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

BACKGROUND = QColor(32, 36, 44)         # same dark slate as the splash screen
PART = (QColor(126, 190, 255), QColor(64, 132, 214), QColor(38, 92, 164))   # top, left, right face of the part
LABEL = QColor(74, 222, 128)            # annotation green (bounding box)
SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw_icon(size: int) -> QPixmap:
    """The icon as a size x size pixmap with a transparent surround."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = size / 64.0                     # the design is laid out on a 64 x 64 grid

    def pt(x, y):
        return QPointF(x * s, y * s)

    # rounded dark tile
    tile = QPainterPath()
    tile.addRoundedRect(QRectF(1 * s, 1 * s, 62 * s, 62 * s), 13 * s, 13 * s)
    p.fillPath(tile, BACKGROUND)

    # the part: an isometric cube (three visible faces, lit from the top)
    cx, cy, w, h = 32.0, 33.0, 14.5, 8.4          # centre, half width, half height of the top rhombus
    d = 15.5                                       # height of the side faces
    top = [pt(cx, cy - d / 2 - h), pt(cx + w, cy - d / 2), pt(cx, cy - d / 2 + h), pt(cx - w, cy - d / 2)]
    left = [pt(cx - w, cy - d / 2), pt(cx, cy - d / 2 + h), pt(cx, cy + d / 2 + h), pt(cx - w, cy + d / 2)]
    right = [pt(cx, cy - d / 2 + h), pt(cx + w, cy - d / 2), pt(cx + w, cy + d / 2), pt(cx, cy + d / 2 + h)]
    p.setPen(Qt.NoPen)
    for face, colour in zip((top, left, right), PART):
        p.setBrush(colour)
        p.drawPolygon(QPolygonF(face))
    if size >= 32:                                 # a stud on top: reads as a building-block / machined part
        p.setBrush(PART[0].lighter(118))
        p.drawEllipse(pt(cx, cy - d / 2 - 0.6), 5.2 * s, 3.0 * s)

    # the label: corner brackets of a bounding box around the part
    pen = QPen(LABEL, max(1.6, 4.2 * s), Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    x0, y0, x1, y1, a = 10.0, 10.0, 54.0, 54.0, 11.0
    for (ax, ay), (dx, dy) in (((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x1, y1), (-1, -1)), ((x0, y1), (1, -1))):
        corner = QPainterPath(pt(ax + dx * a, ay))
        corner.lineTo(pt(ax, ay))
        corner.lineTo(pt(ax, ay + dy * a))
        p.drawPath(corner)
    p.end()
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for n in SIZES:
        icon.addPixmap(draw_icon(n))
    return icon


def write_icon_files(folder: str | Path) -> list:
    """Export syngt.png (256 px) and syngt.ico (all sizes) for shortcuts / launchers. Needs a QGuiApplication."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    png = folder / "syngt.png"
    draw_icon(256).save(str(png), "PNG")
    out = [png]
    try:                                # ICO with every size (Windows shortcuts, installers)
        from PIL import Image
        ico = folder / "syngt.ico"
        Image.open(png).save(ico, sizes=[(n, n) for n in SIZES])
        out.append(ico)
    except Exception:
        pass
    return out
