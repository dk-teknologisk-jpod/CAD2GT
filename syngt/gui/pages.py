"""Configuration pages of the main window."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QGridLayout,
                               QGroupBox, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..assets import MESH_EXTS, guess_file_unit, list_mesh_files, load_asset
from ..cameras import (PRESET_BY_KEY, PRESETS, camera_from_preset, intrinsics_for, is_auto_camera_name,
                       load_calibration, preset_slug, save_calibration_json)
from ..config import ALL_FORMATS, CameraConfig, ObjectModel
from ..project import Project
from .realism import PaletteEdit
from .widgets import Binder, group, hbox, note, scroll_page

MESH_FILTER = "CAD / mesh (" + " ".join(f"*{e}" for e in sorted(MESH_EXTS)) + ")"


def show_rows(group_box, visible: dict) -> None:
    """Show only the rows of a form group that have an effect right now ({row widget: bool})."""
    form = group_box.layout()
    for w, on in visible.items():
        form.setRowVisible(w, bool(on))


def _clear(layout):
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.deleteLater()


# =========================================================================== objects
class ObjectsPage(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        self.project, self.cfg = project, project.cfg
        b = self.b = Binder(project)
        cfg = self.cfg
        oc = cfg.objects
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._item_changed)
        lv.addWidget(QLabel("<b>CAD models</b> (checked = used)"))
        lv.addWidget(self.list, 1)
        bl = QGridLayout()
        for i, (txt, fn) in enumerate([("Add files…", self._add_files), ("Add folder…", self._add_folder),
                                        ("Duplicate", self._dup), ("Remove", self._remove),
                                        ("Inspect mesh", self._inspect), ("Auto class IDs", self._auto_ids),
                                        ("Class names = CAD file names", self._cad_names)]):
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            bl.addWidget(btn, i // 2, i % 2)
        lv.addLayout(bl)
        split.addWidget(left)
        self.detail = QWidget()
        self.detail_l = QVBoxLayout(self.detail)
        self.detail_l.setContentsMargins(0, 0, 0, 0)
        split.addWidget(scroll_page(self.detail, stretch=False))
        split.setStretchFactor(1, 2)

        d = oc.distractors
        total_w = b.irange_w(oc.total_count, 0, 500)
        classes_w = b.ispin(oc, "min_classes_per_scene", 0, 1000)
        count_g = group("Instances per scene", [
            ("Count mode", b.combo(oc, "count_mode", [("per_model", "per part (each part: its own min/max)"),
                                                    ("total", "total (parts drawn by weight)")],
                                   on_change=lambda m: show_rows(count_g, {total_w: m == "total", classes_w: m == "total"}))),
            ("Total instances", total_w),
            ("Min. distinct classes", classes_w),
        ])
        show_rows(count_g, {total_w: oc.count_mode == "total", classes_w: oc.count_mode == "total"})
        dist_g = group("Distractors (rendered, occlude, collide - never annotated)", [
            (None, b.check(d, "enabled", "Enable distractors")),
            ("Count", b.irange_w(d.count, 0, 200)),
            ("Primitive shapes", b.flags(d, "shapes", ["box", "cylinder", "sphere", "capsule", "cone", "torus"])),
            ("Mesh folder (optional)", b.path(d, "mesh_dir", "dir", tip="Random unlabelled meshes, e.g. other parts")),
            ("Mesh folder unit", b.combo(d, "mesh_unit", ["auto", "mm", "cm", "m", "inch"])),
            ("Size (largest dim)", b.range_w(d.size, 0.001, 5, 0.005, 3, "m")),
            ("Colour", b.combo(d.material, "mode", [("random_color", "random colour per distractor"),
                                                    ("palette", "one of the palette colours"),
                                                    ("fixed_color", "one colour"),
                                                    ("original", "from the mesh file (one colour if it has none)")])),
            ("One colour", b.color(d.material, "color", on_pick=lambda: setattr(d.material, "mode", "fixed_color"))),
            ("Palette", PaletteEdit(b, d.material)),
            ("Random colour: max saturation", b.dspin(d.material, "random_saturation_max", 0, 1, 0.05, 2,
                                                    tip="Real clutter is mostly grey/black/white: keep this low")),
            ("Roughness", b.range_w(d.material.roughness, 0, 1, 0.05, 2)),
            ("Metallic", b.range_w(d.material.metallic, 0, 1, 0.05, 2)),
        ])
        v = QVBoxLayout(self)
        v.addWidget(split, 3)
        v.addWidget(scroll_page(count_g, dist_g, stretch=False), 2)
        self._refresh_list()

    # ----------------------------------------------------------------- list
    def _refresh_list(self, select: int = None):
        self.list.blockSignals(True)
        self.list.clear()
        for m in self.cfg.objects.models:
            it = QListWidgetItem(f"[{m.class_id}] {m.class_name}  ({m.unit})")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if m.enabled else Qt.Unchecked)
            it.setToolTip(m.path)
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.cfg.objects.models:
            self.list.setCurrentRow(min(select if select is not None else 0, len(self.cfg.objects.models) - 1))
        else:
            self._select(-1)

    def _item_changed(self, it):
        r = self.list.row(it)
        if 0 <= r < len(self.cfg.objects.models):
            self.b._commit(self.cfg.objects.models[r], "enabled", it.checkState() == Qt.Checked)

    def _next_id(self):
        return max([m.class_id for m in self.cfg.objects.models], default=0) + 1

    def _add_paths(self, paths):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for p in paths:
                # resolve the unit now (files in one folder often mix mm and m) - visible and editable per model
                self.cfg.objects.models.append(ObjectModel(path=str(Path(p).resolve()), name=Path(p).stem,
                                                           class_id=self._next_id(), unit=guess_file_unit(p)))
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_list(len(self.cfg.objects.models) - 1)
        self.b.touch("add parts")

    def _add_files(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add CAD files", "", MESH_FILTER + ";;All (*)")
        if ps:
            self._add_paths(ps)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add all CAD files from folder")
        if d:
            self._add_paths(list_mesh_files(d))

    def _dup(self):
        r = self.list.currentRow()
        if r >= 0:
            m = self.cfg.objects.models[r].model_copy(deep=True)
            m.class_id = self._next_id()
            m.name = m.name + "_copy"
            self.cfg.objects.models.insert(r + 1, m)
            self._refresh_list(r + 1)
            self.b.touch("duplicate part")

    def _remove(self):
        r = self.list.currentRow()
        if r >= 0:
            del self.cfg.objects.models[r]
            self._refresh_list(max(0, r - 1))
            self.b.touch("remove part")

    def _auto_ids(self):
        for i, m in enumerate(self.cfg.objects.models, 1):
            m.class_id = i
        self._refresh_list(self.list.currentRow())
        self.b.touch("auto class IDs")

    def _cad_names(self):
        """Label every part with the name of its CAD file (the default; undoes typed-in names)."""
        with self.project.batch("class names = CAD file names"):
            for m in self.cfg.objects.models:
                m.name = m.cad_name

    def _inspect(self):
        r = self.list.currentRow()
        if r < 0:
            return
        m = self.cfg.objects.models[r]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            a = load_asset(m.path, m.unit, m.recenter, self.cfg.render.smooth_angle_deg, m.name,
                           collision=self.cfg.placement.collision_geometry,
                           max_parts=self.cfg.placement.max_decomposition_parts,
                           stable_max_poses=self.cfg.placement.stable_max_poses,
                           stable_min_prob=self.cfg.placement.stable_min_prob)
            sp = ", ".join(f"{p:.2f}" for _, p in a.stable_poses)
            txt = (f"<b>{a.name}</b><br>file unit guess: <b>{a.detected_unit}</b> (using: {m.unit})<br>"
                   f"extent: {np.round(a.extent * 1000, 2)} mm, diameter {a.diameter * 1000:.1f} mm<br>"
                   f"faces: {a.n_faces}, parts: {len(a.parts)}, texture: {a.has_texture}, vertex colours: {a.has_colors}<br>"
                   f"bbox min (mm): {np.round(a.bbox_min * 1000, 2)}<br>"
                   f"stable poses: {len(a.stable_poses)} (probabilities: {sp})<br>"
                   f"collision hulls: {len(a.collision_hulls)} ({self.cfg.placement.collision_geometry})")
            self.info.setText(txt)
        except Exception as e:
            QMessageBox.warning(self, "Inspect failed", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    # ----------------------------------------------------------------- details
    def _select(self, r):
        _clear(self.detail_l)
        if r < 0 or r >= len(self.cfg.objects.models):
            self.detail_l.addWidget(note("Add CAD files (STL, OBJ, PLY, GLB/GLTF, OFF, 3MF, DAE, STEP). "
                                         "The CAD origin/frame is kept, so the GT pose refers to your CAD frame."))
            return
        m = self.cfg.objects.models[r]
        b = self.b

        def rename(_=None):
            it = self.list.item(r)
            if it:
                it.setText(f"[{m.class_id}] {m.class_name}  ({m.unit})")
        name_w = b.line(m, "name", on_change=rename,
                        tip="Written into every label (BOP, COCO, YOLO). It is the CAD file name unless you type "
                            "another one; leave it empty to use the CAD file name.")
        name_w.setPlaceholderText(m.cad_name)
        file_w = b.path(m, "path", "file", MESH_FILTER)
        was = {"cad": m.cad_name}

        def file_changed(_=None):
            # the class name follows the CAD file, unless the user gave the part a name of their own
            if m.name.strip() in ("", was["cad"]):
                self.project.set(m, "name", m.cad_name, "class name")
                name_w.setText(m.name)
            was["cad"] = m.cad_name
            name_w.setPlaceholderText(m.cad_name)
            rename()
        file_w.changed.connect(file_changed)
        g1 = group("Model", [
            ("File", file_w),
            ("Class name", name_w),
            ("Class ID (BOP obj_id)", b.ispin(m, "class_id", 0, 100000, on_change=rename,
                                            tip="Several models may share a class ID (e.g. variants) for detection")),
            ("File unit", b.combo(m, "unit", [("auto", "auto (guess)"), "mm", "cm", "m", "inch"], on_change=rename,
                                tip="Unit of the numbers in the file. Wrong unit = object 1000x too small/large")),
            (None, b.check(m, "recenter", "Re-centre origin to bbox centre (changes GT frame!)")),
            ("Instances (per_model mode)", b.irange_w(m.count, 0, 500)),
            ("Weight (total mode)", b.dspin(m, "weight", 0, 1000, 0.1, 2)),
            ("Scale jitter", b.range_w(m.scale_jitter, 0.01, 100, 0.01, 3, tip="Keep 1.0 for pose GT of real parts")),
            ("Symmetry", b.combo(m, "symmetry", ["none", "continuous_x", "continuous_y", "continuous_z", "discrete_x",
                                               "discrete_y", "discrete_z", ("box_xyz", "box (180° about x, y, z)")],
                               tip="Written to BOP models_info.json (for symmetry-aware pose losses / metrics)")),
            ("Discrete symmetry order", b.ispin(m, "symmetry_order", 2, 360)),
            ("Density", b.dspin(m, "density", 1, 30000, 100, 0, "kg/m³")),
            ("Friction (object)", b.dspin(m, "friction", 0, 3, 0.05, 2)),
        ])
        g2 = self._material_group(m)
        self.info = QLabel("Press <i>Inspect mesh</i> to load the mesh and show units, size and stable poses.")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_l.addWidget(g1)
        self.detail_l.addWidget(g2)
        self.detail_l.addWidget(group("Mesh info", [(None, self.info)]))
        self.detail_l.addStretch(1)


    def _material_group(self, m: ObjectModel):
        """Colour of one part. Only the rows that have an effect in the chosen mode are shown, and picking a
        colour selects 'one colour' - so a chosen colour is never silently ignored."""
        b, mat = self.b, m.material
        mode = b.combo(mat, "mode", [("original", "from the CAD file (texture / colours)"),
                                     ("fixed_color", "one colour"),
                                     ("palette", "one of several colours (palette), new per instance"),
                                     ("random_color", "random colour per instance (colour-agnostic training)")])
        colour = b.color(mat, "color", on_pick=lambda: setattr(mat, "mode", "fixed_color"))
        hint = note("")
        rows = {
            "colour": ("Colour", colour),
            "palette": ("Palette", PaletteEdit(b, mat)),
            "random": ("Max. saturation", b.dspin(mat, "random_saturation_max", 0, 1, 0.05, 2,
                                                  tip="0 = greys only, 1 = any vivid colour")),
            "hue": ("Hue variation ±", b.dspin(mat, "hue_jitter", 0, 0.5, 0.01, 3)),
            "sat": ("Saturation variation ±", b.dspin(mat, "saturation_jitter", 0, 1, 0.05, 2)),
            "val": ("Brightness variation ±", b.dspin(mat, "value_jitter", 0, 1, 0.05, 2)),
        }
        g = group("Colour && surface", [("Colour comes", mode), (None, hint)] + list(rows.values()) + [
            ("Roughness", b.range_w(mat.roughness, 0, 1, 0.05, 2, tip="0 = glossy, 1 = matt")),
            ("Metallic", b.range_w(mat.metallic, 0, 1, 0.05, 2)),
        ])
        form = g.layout()

        def show(_=None):
            md = mat.mode
            vis = {"colour": md in ("original", "fixed_color"), "palette": md == "palette",
                   "random": md == "random_color", "hue": md != "random_color", "sat": md != "random_color",
                   "val": md != "random_color"}
            for key, (_lab, w) in rows.items():
                form.setRowVisible(w, vis[key])
            hint.setText({
                "original": "Texture or colours stored in the file are used. Files without colour (STL, STEP …) "
                            "are shown in the colour below.",
                "fixed_color": "Every instance gets the colour below (the file's own colours are not used).",
                "palette": "Every instance gets one of the palette colours.",
                "random_color": "Every instance gets a new random colour.",
            }[md])
        mode.currentIndexChanged.connect(show)
        show()
        return g


# =========================================================================== placement
MODE_HELP = {
    "free_floating": "Objects float in a 3D volume with uniformly random 6D rotations (no support plane needed). "
                     "Best for generic detectors / viewpoint-invariant pose estimators.",
    "stable_separated": "Objects rest in their quasi-static stable poses (from the convex hull + COM), randomly "
                        "rotated about z, footprints non-overlapping (min spacing). Optional short physics settle.",
    "physics_drop": "Objects are dropped (MuJoCo) onto the plane. A small cluster radius produces piles where "
                    "objects lean on and occlude each other; a large radius gives scattered, physically resting parts.",
    "physics_bin": "Like physics drop but into a bin (bin picking). Bin walls are rendered as an occluder.",
}


class PlacementPage(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        b = Binder(project)
        p = project.cfg.placement
        self.help = note("")
        mode_g = group("Placement mode", [
            ("Mode", b.combo(p, "mode", [("free_floating", "Free floating (random 6D)"),
                                       ("stable_separated", "Stable poses, separated"),
                                       ("physics_drop", "Physics drop (piles / leaning / scattered)"),
                                       ("physics_bin", "Physics drop into bin")], on_change=self._mode)),
            (None, self.help),
            ("Support area (x, y)", b.vec_w(p, "area_size", 2, 0.01, 20, 0.05, 3, ["x", "y"], "m")),
            ("Yaw", b.range_w(p.yaw, -360, 360, 5, 1, "deg")),
            ("Tilt jitter (max)", b.dspin(p, "tilt_jitter_deg", 0, 90, 1, 1, "deg",
                                        tip="Random tilt around stable/upright pose")),
        ])
        self.g_sep = group("Stable poses / separated", [
            ("Min. stable pose probability", b.dspin(p, "stable_min_prob", 0, 1, 0.01, 3)),
            ("Max. stable poses per object", b.ispin(p, "stable_max_poses", 1, 64)),
            ("Min. spacing between footprints", b.dspin(p, "min_spacing", 0, 1, 0.005, 3, "m")),
            (None, b.check(p, "settle_with_physics", "Settle with short physics simulation (exact contact)")),
        ])
        self.g_phys = group("Physics (MuJoCo)", [
            ("Cluster radius", b.range_w(p.cluster_radius, 0, 5, 0.01, 3, "m",
                                       tip="Spread of the drop positions. Small = piles & leaning")),
            ("Drop height", b.range_w(p.drop_height, 0, 5, 0.01, 3, "m")),
            ("Initial orientation", b.combo(p, "initial_orientation", [("random", "uniform random"),
                                                                     ("stable_pose", "stable pose + tilt"),
                                                                     ("upright", "upright + yaw")])),
            (None, b.check(p, "sequential_drop", "Drop sequentially (one after another -> stacks)")),
            ("Friction", b.range_w(p.friction, 0, 3, 0.05, 2)),
            ("Contact damping ratio", b.dspin(p, "restitution_damping", 0.1, 5, 0.1, 2, tip="1 = critically damped, <1 bouncier")),
            ("Max. sim time", b.dspin(p, "sim_time", 0.1, 60, 0.5, 2, "s")),
            ("Timestep", b.dspin(p, "timestep", 0.0002, 0.01, 0.0005, 4, "s")),
            ("Collision geometry", b.combo(p, "collision_geometry", [("convex_hull", "convex hull (fast)"),
                                                                   ("convex_decomposition", "convex decomposition (CoACD, concave parts)")])),
            ("Max. decomposition parts", b.ispin(p, "max_decomposition_parts", 1, 128)),
            (None, b.check(p, "remove_out_of_area", "Remove objects that rolled out of the area / bin")),
        ])
        self.g_bin = group("Bin", [
            ("Inner size (x, y, height)", b.vec_w(p, "bin_size", 3, 0.01, 10, 0.01, 3, ["x", "y", "h"], "m")),
            ("Wall thickness", b.dspin(p, "bin_wall_thickness", 0.001, 0.2, 0.002, 3, "m")),
            (None, b.check(p, "render_bin", "Render the bin")),
        ])
        self.g_float = group("Free floating", [
            ("Volume (x, y, z)", b.vec_w(p, "float_volume", 3, 0.001, 20, 0.05, 3, ["x", "y", "z"], "m")),
            ("Volume centre", b.vec_w(p, "float_center", 3, -20, 20, 0.05, 3, ["x", "y", "z"], "m")),
            ("Rotation", b.combo(p, "float_rotation", [("uniform_so3", "uniform SO(3)"), ("stable_pose", "stable pose + yaw"),
                                                     ("upright_yaw", "upright + yaw")])),
            (None, b.check(p, "float_avoid_collisions", "Avoid overlaps (bounding spheres)")),
        ])
        v = QVBoxLayout(self)
        v.addWidget(scroll_page(mode_g, self.g_sep, self.g_phys, self.g_bin, self.g_float))
        self._mode(p.mode)

    def _mode(self, m):
        self.help.setText(MODE_HELP.get(m, ""))
        self.g_sep.setVisible(m == "stable_separated" or m in ("physics_drop", "physics_bin"))
        self.g_phys.setVisible(m in ("physics_drop", "physics_bin", "stable_separated"))
        self.g_bin.setVisible(m == "physics_bin")
        self.g_float.setVisible(m == "free_floating")


# =========================================================================== scene
class ScenePage(QWidget):
    def __init__(self, project: Project, open_rectifier):
        super().__init__()
        cfg = project.cfg
        g, bg, l = cfg.scene.ground, cfg.scene.background, cfg.scene.lighting
        b = Binder(project)
        self.tex_path = b.path(g, "texture_dir", "dir", tip="Images; a sidecar .json with width_m/height_m gives metric scale")
        rect_btn = QPushButton("Phone photo → metric ground texture…")
        rect_btn.clicked.connect(lambda: open_rectifier(self.tex_path))
        from .realism import PaletteEdit, table_colours_dialog
        tc_btn = QPushButton("Table colours from real photos…")
        tc_btn.clicked.connect(lambda: table_colours_dialog(self, project))
        g_g = group("Ground plane (support surface, receives shadows)", [
            (None, b.check(g, "enabled", "Ground plane enabled (physics + rendering)")),
            ("Visible probability", b.dspin(g, "visible_probability", 0, 1, 0.05, 2,
                                          tip="<1: sometimes hide the plane (objects over 2D background)")),
            ("Texture mode", b.combo(g, "texture_mode", [("mixed", "mixed"), ("solid_random", "solid random colour"),
                                                       ("procedural", "procedural (noise/wood/checker/…)"),
                                                       ("image_folder", "image folder / photo textures"),
                                                       ("shadow_catcher", "invisible – only shadows, on the background photo"),
                                                       ("palette", "table colours (palette, e.g. measured from photos)")])),
            ("Max colour saturation", b.dspin(g, "saturation_max", 0, 1, 0.05, 2,
                                            tip="Random / procedural table colours. Real tables: ~0.1-0.2")),
            ("Shadow-catcher darkness", b.range_w(g.catcher_strength, 0, 1, 0.05, 2)),
            ("Texture folder", self.tex_path),
            (None, rect_btn),
            ("Table colours (mode 'palette')", PaletteEdit(b, g, "texture_mode")),
            (None, tc_btn),
            ("Texture tile size", b.range_w(g.texture_size_m, 0.01, 20, 0.05, 3, "m", tip="If no metric sidecar is present")),
            ("Texture rotation", b.range_w(g.rotation_deg, -360, 360, 5, 1, "deg")),
            ("Roughness", b.range_w(g.roughness, 0, 1, 0.05, 2)),
            ("Plane half extent", b.dspin(g, "extent_m", 0.1, 100, 0.5, 2, "m")),
        ])
        b_g = group("Background (behind the plane / when no plane)", [
            ("Mode", b.combo(bg, "mode", [("mixed", "mixed"), ("solid_random", "solid random colour"), ("gradient", "gradient"),
                                       ("noise", "procedural noise"), ("image_folder", "image folder (e.g. your phone photos)"),
                                       ("transparent", "transparent (RGBA PNG)")])),
            ("Image folder", b.path(bg, "image_dir", "dir")),
            (None, b.check(bg, "random_crop", "Random crop")),
            ("Crop size", b.range_w(bg.crop_scale, 0.05, 1, 0.05, 2, tip="Fraction of the photo kept. Photos from your own "
                                                                       "camera: 0.85-1 keeps the true scale")),
            (None, b.check(bg, "flip", "Random horizontal flip")),
            ("Brightness", b.range_w(bg.brightness, 0, 5, 0.05, 2)),
            ("Max colour saturation", b.dspin(bg, "saturation_max", 0, 1, 0.05, 2, tip="Random backdrops (photos are kept)")),
            (None, note("Tip: put photos of your real workcell into the image folder for backdrops, and rectified "
                        "photos of the real table surface into the ground texture folder for physically scaled, "
                        "shadow-receiving support surfaces.")),
        ])
        l_g = group("Lighting", [
            ("Number of lights", b.irange_w(l.num_lights, 0, 8)),
            ("Light type", b.combo(l, "light_type", ["mixed", "point", "directional"])),
            ("Irradiance at scene", b.range_w(l.intensity, 0, 1000, 0.5, 2, tip="Point lights are scaled by d² to give this irradiance at the scene centre")),
            ("Colour temperature", b.range_w(l.color_temperature, 1000, 20000, 100, 0, "K")),
            ("Distance (point)", b.range_w(l.distance, 0.05, 50, 0.1, 2, "m")),
            ("Elevation", b.range_w(l.elevation_deg, -90, 90, 5, 1, "deg")),
            ("Ambient (hemisphere)", b.range_w(l.ambient, 0, 10, 0.05, 2)),
            (None, b.check(l, "shadows", "Shadows (first light)")),
            ("Shadow softness", b.range_w(l.shadow_softness, 0.5, 40, 0.5, 1,
                                        tip="Penumbra size. 1 = hard spot light, 8-15 = diffuse ceiling / area light")),
            ("Shadow map size", b.combo(l, "shadow_map_size", [1024, 2048, 4096])),
            (None, b.check(l, "relight_per_view", "Re-sample lights for every view")),
        ])
        v = QVBoxLayout(self)
        v.addWidget(scroll_page(g_g, b_g, l_g))


# =========================================================================== cameras
class CamerasPage(QWidget):
    changed = Signal()

    def __init__(self, project: Project):
        super().__init__()
        self.project, self.cfg = project, project.cfg
        self.b = Binder(project)
        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("<b>Cameras</b> (multi-view = several)"))
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        lv.addWidget(self.list, 1)
        # preset chooser
        pg = QGroupBox("Add from preset")
        pl = QVBoxLayout(pg)
        self.preset = QComboBox()
        for p in PRESETS:
            self.preset.addItem(p.key)
        self.stream = QComboBox()
        self.stream.addItems(["color", "depth"])
        self.res = QComboBox()
        self.preset.currentIndexChanged.connect(self._fill_res)
        self.stream.currentIndexChanged.connect(self._fill_res)
        self.preset_note = note("")
        pl.addWidget(self.preset)
        pl.addWidget(hbox(QLabel("stream"), self.stream, QLabel("mode"), self.res))
        pl.addWidget(self.preset_note)
        b_add = QPushButton("Add camera")
        b_add.clicked.connect(self._add_preset)
        b_apply = QPushButton("Apply to selected")
        b_apply.clicked.connect(self._apply_preset)
        pl.addWidget(hbox(b_add, b_apply))
        lv.addWidget(pg)
        bl = QGridLayout()
        for i, (txt, fn) in enumerate([("Import calibration…", self._import), ("Export calibration…", self._export),
                                        ("Duplicate", self._dup), ("Remove", self._remove)]):
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            bl.addWidget(btn, i // 2, i % 2)
        lv.addLayout(bl)
        lv.addWidget(note("Calibration import: rs-enumerate-devices/pyrealsense JSON, OpenCV YAML/XML, ROS camera_info, "
                          "Kalibr camchain (multi-camera rig), plain JSON {fx,fy,cx,cy,width,height,dist,T_world_cam}."))
        split.addWidget(left)
        self.detail = QWidget()
        self.detail_l = QVBoxLayout(self.detail)
        self.detail_l.setContentsMargins(0, 0, 0, 0)
        split.addWidget(scroll_page(self.detail, stretch=False))
        split.setStretchFactor(1, 2)
        QVBoxLayout(self).addWidget(split)
        self._fill_res()
        self._refresh(notify=False)

    def _fill_res(self):
        p = PRESET_BY_KEY[self.preset.currentText()]
        spec = p.color if self.stream.currentText() == "color" else p.depth
        self.res.clear()
        for w, h in spec.resolutions:
            self.res.addItem(f"{w}x{h}", (w, h))
        fx, fy, cx, cy = intrinsics_for(spec, *spec.native)
        self.preset_note.setText(f"{p.vendor} {p.model}: {self.stream.currentText()} native {spec.native[0]}x{spec.native[1]}, "
                                 f"HFOV {spec.hfov}°, VFOV {spec.vfov}° (fx≈{fx:.0f}px). Depth: {p.depth_type}, baseline "
                                 f"{p.baseline_m * 1000:.0f} mm, range {p.min_z}-{p.max_z} m. {p.notes} "
                                 "Nominal values from datasheet FOVs - import your unit's calibration for exact intrinsics.")

    def _preset_cam(self, name):
        w, h = self.res.currentData()
        return camera_from_preset(self.preset.currentText(), self.stream.currentText(), w, h, name=name)

    def _refresh(self, select=None, notify=True):
        self.list.blockSignals(True)
        self.list.clear()
        for c in self.cfg.cameras:
            self.list.addItem(f"{c.name}  {c.width}x{c.height}  [{c.preset}]")
        self.list.blockSignals(False)
        n = len(self.cfg.cameras)
        if n:
            self.list.setCurrentRow(min(select if select is not None else 0, n - 1))
        else:
            self._select(-1)
        self.changed.emit()
        if notify:  # camera added / replaced / removed -> preview, quick-setup read-out, dirty flag
            self.b.touch("cameras")

    def _unique_name(self, base: str, skip: int = -1) -> str:
        from ..writers import safe_name
        used = {safe_name(c.name) for i, c in enumerate(self.cfg.cameras) if i != skip}
        name, k = base, 1
        while safe_name(name) in used:
            k += 1
            name = f"{base}_{k}"
        return name

    def _add_preset(self):
        self.cfg.cameras.append(self._preset_cam(self._unique_name(preset_slug(self.preset.currentText()))))
        self._refresh(len(self.cfg.cameras) - 1)

    def _apply_preset(self):
        r = self.list.currentRow()
        if r < 0:
            return
        old = self.cfg.cameras[r]
        # auto-generated names follow the preset (they end up in file names); user-typed names are kept
        name = self._unique_name(preset_slug(self.preset.currentText()), skip=r) if is_auto_camera_name(old.name)             else old.name
        c = self._preset_cam(name)
        # a preset defines the optics (resolution, intrinsics, depth sensor type); everything the user set up
        # for this camera in the project stays
        c.rig_translation, c.rig_rotation_deg, c.world_pose = old.rig_translation, old.rig_rotation_deg, old.world_pose
        c.intrinsics_jitter = old.intrinsics_jitter
        c.depth.enabled = old.depth.enabled
        self.cfg.cameras[r] = c
        self._refresh(r)

    def _import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import calibration", "", "Calibration (*.json *.yaml *.yml *.xml);;All (*)")
        if not p:
            return
        try:
            cams = load_calibration(p)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        if not cams:
            QMessageBox.information(self, "Import", "The file contains no camera.")
            return
        r = self.list.currentRow()
        replace = False
        if len(cams) == 1 and r >= 0:
            box = QMessageBox(QMessageBox.Question, "Import calibration",
                              f"Use the calibration for the selected camera '{self.cfg.cameras[r].name}' "
                              f"(keeps its name, rig pose and depth settings), or add it as a new camera?", parent=self)
            b_rep = box.addButton("Replace selected", QMessageBox.AcceptRole)
            box.addButton("Add as new camera", QMessageBox.ActionRole)
            b_cancel = box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is b_cancel or box.clickedButton() is None:
                return
            replace = box.clickedButton() is b_rep
        if replace:
            c, old = cams[0], self.cfg.cameras[r]
            c.name = old.name
            c.rig_translation, c.rig_rotation_deg = old.rig_translation, old.rig_rotation_deg
            c.world_pose = c.world_pose or old.world_pose
            c.depth = old.depth
            c.intrinsics_jitter = old.intrinsics_jitter
            self.cfg.cameras[r] = c
            self._refresh(r)
        else:
            for c in cams:
                c.name = self._unique_name(c.name)
                self.cfg.cameras.append(c)
            self._refresh(len(self.cfg.cameras) - 1)

    def _export(self):
        p, _ = QFileDialog.getSaveFileName(self, "Export calibration", "cameras.json", "JSON (*.json)")
        if p:
            save_calibration_json(self.cfg.cameras, p)

    def _dup(self):
        r = self.list.currentRow()
        if r >= 0:
            c = self.cfg.cameras[r].model_copy(deep=True)
            c.name = self._unique_name(c.name + "_copy")     # the name is part of every file name
            self.cfg.cameras.insert(r + 1, c)
            self._refresh(r + 1)

    def _remove(self):
        r = self.list.currentRow()
        if r >= 0 and len(self.cfg.cameras) > 1:
            del self.cfg.cameras[r]
            self._refresh(max(0, r - 1))

    def _select(self, r):
        _clear(self.detail_l)
        if r < 0:
            return
        c: CameraConfig = self.cfg.cameras[r]
        dp = c.depth
        b = self.b

        def upd(_=None):
            it = self.list.item(r)
            if it:
                it.setText(f"{c.name}  {c.width}x{c.height}  [{c.preset}]")
            self.changed.emit()

        fov = QLabel()

        def upd_fov(_=None):
            hf = 2 * math.degrees(math.atan(c.width / 2 / max(c.fx, 1e-6)))
            vf = 2 * math.degrees(math.atan(c.height / 2 / max(c.fy, 1e-6)))
            fov.setText(f"HFOV {hf:.1f}°, VFOV {vf:.1f}°")
            upd()

        new_w = b.ispin(c, "width", 8, 16384, on_change=upd_fov)
        new_h = b.ispin(c, "height", 8, 16384, on_change=upd_fov)
        rescale = QPushButton("Rescale intrinsics…")
        rescale.setToolTip("Scale fx, fy, cx, cy from a previous resolution to the current width/height")
        rescale.clicked.connect(lambda: self._rescale(c, r))
        self.world_txt = QPlainTextEdit(json.dumps(c.world_pose) if c.world_pose else "")
        self.world_txt.setPlaceholderText("T_world_cam 4x4 (JSON) for viewpoint mode 'fixed'; empty = default")
        self.world_txt.setMaximumHeight(70)

        def set_world():
            t = self.world_txt.toPlainText().strip()
            try:
                pose = np.asarray(json.loads(t), float).reshape(4, 4).tolist() if t else None
                self.world_txt.setStyleSheet("")
            except Exception:
                self.world_txt.setStyleSheet("background:#a33;")   # not valid yet: keep the last valid pose
                return
            b._commit(c, "world_pose", pose)
        self.world_txt.textChanged.connect(set_world)
        g1 = group("Intrinsics (OpenCV pinhole + Brown-Conrady)", [
            ("Name", b.line(c, "name", on_change=upd)),
            ("Preset / source", QLabel(c.preset + (f"  ({Path(c.calibration_file).name})" if c.calibration_file else ""))),
            ("Resolution", hbox(new_w, QLabel("x"), new_h, rescale)),
            ("fx, fy", hbox(b.dspin(c, "fx", 1, 1e6, 1, 3, on_change=upd_fov), b.dspin(c, "fy", 1, 1e6, 1, 3, on_change=upd_fov))),
            ("cx, cy", hbox(b.dspin(c, "cx", -1e5, 1e5, 1, 3), b.dspin(c, "cy", -1e5, 1e5, 1, 3))),
            ("FOV", fov),
            ("Distortion k1 k2 p1 p2", b.vec_w(c, "dist", 4, -10, 10, 0.001, 6)),
            ("Distortion k3 [k4 k5 k6]", b.vec_w(c, "dist", 4, -100, 100, 0.001, 6, start=4,
                                                 tip="k4-k6 = rational model (8 coefficients); leave 0 if unused")),
            ("Intrinsics jitter ±", b.dspin(c, "intrinsics_jitter", 0, 0.5, 0.005, 3,
                                          tip="Relative random perturbation of fx, fy, cx, cy per image (calibration robustness)")),
        ])
        upd_fov()
        g2 = group("Extrinsics", [
            ("Rig translation (x,y,z)", b.vec_w(c, "rig_translation", 3, -100, 100, 0.005, 4, ["x", "y", "z"], "m")),
            ("Rig rotation (xyz euler)", b.vec_w(c, "rig_rotation_deg", 3, -360, 360, 1, 3, ["rx", "ry", "rz"], "°")),
            (None, note("T_rig_cam: pose of this camera in the rig frame (= OpenCV frame of the sampled viewpoint). "
                        "Camera 0 normally stays at identity. Used for rigid multi-camera rigs.")),
            ("World pose (fixed mode)", self.world_txt),
        ])
        g3 = group("Depth sensor model (depth aligned to this camera)", [
            (None, b.check(dp, "enabled", "Render depth")),
            ("Model", b.combo(dp, "model", [("ideal", "ideal (GT)"), ("stereo", "active stereo (RealSense/Gemini)"),
                                          ("tof", "time-of-flight (Femto/L515)")])),
            ("Stereo baseline", b.dspin(dp, "baseline_m", 0.001, 1, 0.005, 4, "m")),
            ("Disparity noise (RMS)", b.dspin(dp, "disparity_noise_px", 0, 5, 0.01, 3, "px", tip="σ_z = z²·σ_d / (f·B)")),
            ("ToF noise rel / abs", hbox(b.dspin(dp, "tof_noise_rel", 0, 0.1, 0.0005, 4), b.dspin(dp, "tof_noise_abs", 0, 0.1, 0.0005, 4, "m"))),
            ("Range min / max", hbox(b.dspin(dp, "min_z", 0, 100, 0.05, 3, "m"), b.dspin(dp, "max_z", 0, 100, 0.5, 2, "m"))),
            ("Edge dropout / flying px", b.dspin(dp, "edge_dropout", 0, 1, 0.05, 2)),
            ("Edge threshold (rel.)", b.dspin(dp, "edge_threshold", 0, 1, 0.005, 3)),
            ("Random holes", b.dspin(dp, "random_dropout", 0, 0.5, 0.001, 4)),
        ])
        self.detail_l.addWidget(g1)
        self.detail_l.addWidget(g2)
        self.detail_l.addWidget(g3)
        self.detail_l.addStretch(1)

    def _rescale(self, c: CameraConfig, r):
        from PySide6.QtWidgets import QInputDialog
        txt, ok = QInputDialog.getText(self, "Rescale intrinsics", "Resolution the current fx/fy/cx/cy belong to (WxH):",
                                       text=f"{int(round(c.cx * 2 + 1))}x{int(round(c.cy * 2 + 1))}")
        if not ok:
            return
        try:
            w0, h0 = [int(x) for x in txt.lower().split("x")]
        except Exception:
            return
        if w0 <= 0 or h0 <= 0:
            return
        s = max(c.width / w0, c.height / h0)
        with self.project.batch("rescale intrinsics"):
            c.fx *= s
            c.fy *= s
            c.cx = (c.cx + 0.5) * s - 0.5 - (w0 * s - c.width) / 2
            c.cy = (c.cy + 0.5) * s - 0.5 - (h0 * s - c.height) / 2


# =========================================================================== viewpoints
VIEW_HELP = {
    "sphere_random": "Independent random viewpoints, uniform on the spherical cap given by elevation/azimuth ranges.",
    "sphere_fibonacci": "Evenly spread (Fibonacci) viewpoints over the cap, random phase per scene - best coverage.",
    "grid": "Deterministic azimuth x elevation grid (views/scene = steps product) - exhaustive permutations.",
    "orbit": "Turntable: fixed elevation & distance per scene, azimuth stepping over the range (sequence-like).",
    "fixed": "Each camera at its calibrated world pose (Cameras page) + optional jitter. Use for a fixed workcell.",
}


class ViewpointsPage(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        b = Binder(project)
        v = project.cfg.viewpoints
        self.help = note("")
        self.v = v
        self.views_w = b.ispin(v, "views_per_scene", 1, 10000)
        g = group("Viewpoint sampling", [
            ("Mode", b.combo(v, "mode", [("sphere_random", "random on sphere cap"), ("sphere_fibonacci", "Fibonacci (even coverage)"),
                                       ("grid", "grid (azimuth x elevation)"), ("orbit", "orbit / turntable"),
                                       ("fixed", "fixed calibrated poses + jitter")], on_change=self._mode)),
            (None, self.help),
            ("Views per scene", self.views_w),
            ("Rig mode", b.combo(v, "rig_mode", [("rigid", "rigid rig (cameras move together, rig extrinsics)"),
                                               ("independent", "independent (each camera own viewpoint)")])),
            ("Look-at target", b.combo(v, "target", [("scene_center", "scene centre"), ("origin", "world origin"),
                                                   ("random_object", "random object")])),
            ("Look-at jitter (σ)", b.dspin(v, "look_at_jitter", 0, 5, 0.005, 3, "m")),
        ])
        self.fill_w = b.range_w(v.fill_fraction, 0.01, 10, 0.05, 2,
                                tip="Angular radius of the scene bounding sphere relative to the half FOV. >1 crops (truncation).")
        self.radius_w = b.range_w(v.radius, 0.01, 100, 0.05, 3, "m")
        g2 = self.g_dist = group("Distance", [
            ("Distance mode", b.combo(v, "distance_mode", [("absolute", "absolute distance"),
                                                         ("fill_fraction", "fill fraction (scene size / FOV)")],
                                      on_change=lambda _: self._mode(v.mode))),
            ("Fill fraction", self.fill_w),
            ("Distance (absolute)", self.radius_w),
        ])
        self.grid_w = hbox(b.ispin(v, "grid_azimuth_steps", 1, 1000), b.ispin(v, "grid_elevation_steps", 1, 1000))
        g3 = self.g_angles = group("Angles (world z-up; elevation 90° = top-down)", [
            ("Elevation", b.range_w(v.elevation_deg, -90, 90, 5, 1, "deg", dist=False)),
            ("Azimuth", b.range_w(v.azimuth_deg, -720, 720, 5, 1, "deg", dist=False)),
            ("In-plane roll", b.range_w(v.roll_deg, -180, 180, 5, 1, "deg")),
            ("Grid azimuth / elevation steps", self.grid_w),
        ])
        self.g_fixed = group("Fixed-pose jitter", [
            ("Translation jitter (σ)", b.dspin(v, "fixed_jitter_translation", 0, 5, 0.005, 3, "m")),
            ("Rotation jitter (max)", b.dspin(v, "fixed_jitter_rotation_deg", 0, 180, 0.5, 2, "deg")),
        ])
        self.g_accept = g4 = group("View acceptance", [
            ("Min. objects in view", b.ispin(v, "min_visible_objects", 0, 1000, tip="Re-sample views with fewer object centres in the image")),
            ("Max. attempts", b.ispin(v, "max_view_attempts", 1, 1000)),
        ])
        self.g_main = g
        QVBoxLayout(self).addWidget(scroll_page(g, g2, g3, self.g_fixed, g4))
        self._mode(v.mode)

    def _mode(self, m):
        """Only the settings that the chosen mode uses are shown."""
        v = self.v
        self.help.setText(VIEW_HELP.get(m, ""))
        fixed, grid = m == "fixed", m == "grid"
        self.g_dist.setVisible(not fixed)
        self.g_angles.setVisible(not fixed)
        self.g_fixed.setVisible(fixed)
        self.g_accept.setVisible(not fixed and not grid)
        show_rows(self.g_main, {self.views_w: not grid})
        show_rows(self.g_dist, {self.fill_w: v.distance_mode == "fill_fraction", self.radius_w: v.distance_mode == "absolute"})
        show_rows(self.g_angles, {self.grid_w: grid})


# =========================================================================== sensor / render
class SensorPage(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        b = Binder(project)
        s, r = project.cfg.sensor, project.cfg.render
        meter_w = b.combo(s, "ae_metering", [
            ("light", "scene light (parts keep their brightness)"),
            ("frame", "whole frame (background changes part brightness)")],
            tip="light: an 18 % grey card at the scene centre is shown at the target brightness, "
                "so a bright or dark table does not darken/brighten the parts.\n"
                "frame: the mean brightness of the whole image is the target, like a camera's average metering.")
        target_w = b.range_w(s.ae_target, 0.05, 0.95, 0.02, 2)
        exposure_w = b.range_w(s.exposure, 0.01, 100, 0.05, 3,
                               tip="Fixed exposure gain. With auto exposure on it is only used for parts rendered onto "
                                   "real background photos (those keep the photo's own exposure).")

        def exposure_rows(_=None):
            show_rows(g, {meter_w: s.auto_exposure, target_w: s.auto_exposure})
        g = group("Colour sensor simulation (applied in linear space, then tone-mapped)", [
            (None, b.check(s, "enabled", "Enable sensor effects")),
            (None, b.check(s, "auto_exposure", "Auto exposure", on_change=exposure_rows,
                         tip="Expose every image to the target brightness (otherwise the fixed Exposure range is used)")),
            ("Auto exposure metering", meter_w),
            ("Auto exposure target", target_w),
            ("Exposure (fixed)", exposure_w),
            ("White balance (R, G, B gain)", b.vec_w(s, "white_balance_bias", 3, 0.3, 3.0, 0.01, 3, ["R", "G", "B"],
                                                     tip="Colour cast of the camera (1, 1, 1 = neutral). Set by a "
                                                         "camera look profile.")),
            ("White balance jitter ±", b.dspin(s, "white_balance_jitter", 0, 1, 0.01, 3)),
            ("Vignetting", b.range_w(s.vignetting, 0, 1, 0.05, 2)),
            ("Tone mapping", b.combo(s, "tonemap", [("filmic", "filmic (ACES-like)"), ("srgb", "clip + sRGB gamma")])),
            ("Defocus probability", b.dspin(s, "defocus_prob", 0, 1, 0.05, 2)),
            ("Defocus σ", b.range_w(s.defocus_sigma, 0, 20, 0.1, 2, "px")),
            ("Motion blur probability", b.dspin(s, "motion_blur_prob", 0, 1, 0.05, 2)),
            ("Motion blur length", b.range_w(s.motion_blur_length, 1, 100, 1, 1, "px")),
            ("Read noise σ", b.range_w(s.gaussian_noise, 0, 1, 0.002, 4)),
            ("Shot noise k", b.range_w(s.shot_noise, 0, 1, 0.002, 4)),
            ("Saturation", b.range_w(s.saturation, 0, 3, 0.05, 2)),
            ("Contrast", b.range_w(s.contrast, 0, 3, 0.05, 2)),
            ("JPEG quality (artefacts)", b.range_w(s.jpeg_quality, 1, 100, 1, 0)),
        ])
        g2 = group("Renderer", [
            ("MSAA samples", b.combo(r, "msaa_samples", [0, 2, 4, 8, 16])),
            ("Near / far clip", hbox(b.dspin(r, "near", 0.0001, 10, 0.005, 4, "m"), b.dspin(r, "far", 0.1, 10000, 1, 2, "m"))),
            ("Canvas padding", b.dspin(r, "canvas_padding", 0, 2, 0.05, 2,
                                     tip="Extra render border: amodal boxes / truncation of objects leaving the image")),
            ("Shading crease angle", b.dspin(r, "smooth_angle_deg", 0, 180, 5, 1, "deg", tip="CAD meshes: split normals at sharper edges")),
            (None, b.check(r, "gpu_postprocess", "Sensor simulation on the GPU (off = CPU reference implementation)")),
        ])
        exposure_rows()
        QVBoxLayout(self).addWidget(scroll_page(g, g2))


# =========================================================================== output
FORMAT_HELP = {
    "native": "rich per-image JSON (always written)",
    "bop": "BOP (6D pose; multi-camera = BOP-H3 style suffixes)",
    "coco": "COCO instances (+OBB, keypoints, pose fields)",
    "coco_bb": "COCO boxes in train/ + validation/ folders (RT-DETR trainer)",
    "yolo": "YOLO detect (axis-aligned boxes)",
    "yolo_obb": "YOLO-OBB (oriented boxes, 4 corners)",
    "yolo_seg": "YOLO-seg (instance polygons)",
    "yolo_pose": "YOLO-pose (projected 3D-box keypoints)",
}


class OutputPage(QWidget):
    def __init__(self, project: Project):
        super().__init__()
        b = Binder(project)
        o, gen = project.cfg.output, project.cfg.generation
        fw = b.flags(o, "formats", [(f, f"{f} - {FORMAT_HELP[f]}") for f in ALL_FORMATS], columns=2,
                     locked=("native",))
        g = group("Output", [
            ("Output folder", b.path(o, "output_dir", "dir")),
            ("Formats", fw),
            ("RGB format", hbox(b.combo(o, "rgb_format", ["jpg", "png"]), QLabel("JPEG quality"), b.ispin(o, "jpg_quality", 1, 100))),
            (None, hbox(b.check(o, "save_depth", "Depth (uint16 PNG)"), QLabel("depth scale"),
                        b.dspin(o, "depth_scale_mm", 0.001, 100, 0.05, 3, "mm/unit"))),
            (None, hbox(b.check(o, "save_instance_masks", "Instance masks"), b.check(o, "save_semantic_masks", "Semantic masks"),
                        b.check(o, "save_normals", "Normals"), b.check(o, "save_amodal_masks", "Amodal masks"), 1)),
        ])
        g2 = group("Label filters & geometry (detector exports; native/BOP keep everything)", [
            ("Min. visible fraction", b.dspin(o, "min_visib_fract", 0, 1, 0.05, 3)),
            ("Min. visible pixels", b.ispin(o, "min_px_visib", 0, 10**7)),
            ("Max. truncation", b.dspin(o, "max_truncation", 0, 1, 0.05, 3)),
            ("OBB source", b.combo(o, "obb_source", [("visible_mask", "min-area rect of visible mask"),
                                                   ("amodal_mask", "min-area rect of amodal silhouette"),
                                                   ("projected_3d_box", "min-area rect of projected 3D box")])),
            ("Keypoints", b.combo(o, "keypoints", [("bbox3d_9", "3D box: centre + 8 corners"), ("bbox3d_8", "3D box: 8 corners")])),
            ("Polygon tolerance", b.dspin(o, "polygon_tolerance_px", 0, 20, 0.5, 2, "px")),
            ("Split train/val/test (by scene)", b.vec_w(o, "split_ratios", 3, 0, 1, 0.05, 3, ["train", "val", "test"])),
        ])
        g3 = group("Generation", [
            ("Number of scenes", b.ispin(gen, "num_scenes", 1, 10**8)),
            ("First scene index", b.ispin(gen, "start_scene", 0, 10**8, tip="Scenes are deterministic by (seed, index): extend datasets by starting later")),
            ("Seed", b.ispin(gen, "seed", 0, 2**31 - 1)),
            ("Worker processes", b.ispin(gen, "workers", 0, 64, tip="auto = all CPU cores but one", special="auto")),
        ])
        self.summary = note("")
        QVBoxLayout(self).addWidget(scroll_page(g, g2, g3, self.summary))
