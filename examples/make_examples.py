"""Create a few example CAD parts (STL, millimetres) for trying the generator."""
import math
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon

out = Path(__file__).parent / "models"
out.mkdir(exist_ok=True)


def gear(n=14, r_out=30.0, r_in=24.0, hole=8.0, h=10.0):
    pts = []
    for i in range(n * 4):
        a = 2 * math.pi * i / (n * 4)
        r = r_out if (i % 4) in (1, 2) else r_in
        pts.append((r * math.cos(a), r * math.sin(a)))
    poly = Polygon(pts).difference(Point(0, 0).buffer(hole, 32))
    return trimesh.creation.extrude_polygon(poly, h)


def bracket():
    a = trimesh.creation.box(extents=[60, 30, 6]); a.apply_translation([30, 15, 3])
    b = trimesh.creation.box(extents=[6, 30, 40]); b.apply_translation([3, 15, 20])
    m = trimesh.boolean.union([a, b], engine="manifold")
    for x in (25, 45):
        c = trimesh.creation.cylinder(radius=3.5, height=20, sections=32); c.apply_translation([x, 15, 3])
        m = trimesh.boolean.difference([m, c], engine="manifold")
    return m


def flange():
    tube = trimesh.creation.annulus(r_min=12, r_max=16, height=40, sections=64); tube.apply_translation([0, 0, 20])
    disk = trimesh.creation.annulus(r_min=12, r_max=32, height=6, sections=64); disk.apply_translation([0, 0, 3])
    m = trimesh.boolean.union([tube, disk], engine="manifold")
    for k in range(4):
        a = k * math.pi / 2
        c = trimesh.creation.cylinder(radius=3, height=20, sections=24); c.apply_translation([24 * math.cos(a), 24 * math.sin(a), 3])
        m = trimesh.boolean.difference([m, c], engine="manifold")
    return m


def hexnut(s=24.0, h=12.0, hole=7.0):
    r = s / math.sqrt(3)
    poly = Polygon([(r * math.cos(math.pi / 3 * i), r * math.sin(math.pi / 3 * i)) for i in range(6)])
    return trimesh.creation.extrude_polygon(poly.difference(Point(0, 0).buffer(hole, 32)), h)


def brick(l=50.0, w=10.0, h=5.0):
    """Plain 50 x 10 x 5 mm brick, CAD origin at its centre."""
    return trimesh.creation.box(extents=[l, w, h])


for name, m in [("brick_50x10x5", brick()), ("gear", gear()), ("bracket", bracket()), ("flange", flange()), ("hexnut", hexnut())]:
    m.export(out / f"{name}.stl")
    print(name, m.is_watertight, m.extents)
