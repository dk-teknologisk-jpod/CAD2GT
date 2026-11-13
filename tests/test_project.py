"""Application model (syngt.project): change tracking, undo, bulk-action reports, session, pre-flight checks,
dataset status. No Qt, no rendering.   Run:  python -m pytest tests/test_project.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syngt.config import ObjectModel, ProjectConfig  # noqa: E402
from syngt.project import BULK, EDIT, RESET, Project, diff_settings  # noqa: E402
from syngt.scene import _material, effective_material_mode  # noqa: E402
from syngt.validation import ERROR, WARNING  # noqa: E402

BRICK = str(ROOT / "examples" / "models" / "brick_50x10x5.stl")


def project_with_brick() -> Project:
    p = Project()
    p.cfg.objects.models.append(ObjectModel(path=BRICK, name="brick", unit="mm"))
    p.touch("add part")
    return p


def test_set_notifies_only_real_changes():
    p = Project()
    events = []
    p.subscribe(events.append)
    assert p.set(p.cfg.generation, "num_scenes", p.cfg.generation.num_scenes) is False
    assert events == [] and not p.dirty
    assert p.set(p.cfg.generation, "num_scenes", 7) is True
    assert events == [EDIT] and p.dirty and p.cfg.generation.num_scenes == 7


def test_touch_without_change_is_silent():
    p = Project()
    events = []
    p.subscribe(events.append)
    assert p.touch("nothing") is False and events == []


def test_bulk_action_reports_exactly_what_it_changed():
    p = Project()
    events = []
    p.subscribe(events.append)
    changed = p.apply_layout_profile("pile")
    assert events == [BULK]
    assert changed and all(k.startswith(("placement.", "scene.ground.enabled")) for k in changed), changed
    # nothing outside the profile's subject is touched
    assert not any(k.startswith(("objects", "sensor", "cameras", "output", "viewpoints")) for k in changed)


def test_undo_redo_roundtrip():
    p = project_with_brick()
    before = p.cfg.to_dict()
    p.apply_view_profile("top")
    after = p.cfg.to_dict()
    assert before != after
    events = []
    p.subscribe(events.append)
    assert p.undo().startswith("Camera position")
    assert p.cfg.to_dict() == before and events == [RESET]
    assert p.redo().startswith("Camera position")
    assert p.cfg.to_dict() == after


def test_edits_of_one_field_collapse_into_one_undo_step():
    p = Project()
    n0 = p.cfg.generation.num_scenes
    for v in (5, 6, 7, 8):          # e.g. dragging a slider
        p.set(p.cfg.generation, "num_scenes", v)
    p.undo()
    assert p.cfg.generation.num_scenes == n0


def test_session_roundtrip_keeps_paths_and_state(tmp_path):
    p = project_with_brick()
    p.cfg.output.output_dir = "output/my_set"        # relative: must NOT be re-based onto the session folder
    p.apply_layout_profile("pile")
    f = tmp_path / "cfgdir" / "session.yaml"
    p.save_session(f)
    q = Project()
    assert q.restore_session(f)
    assert q.cfg.to_dict() == p.cfg.to_dict()
    assert q.cfg.output.output_dir == "output/my_set"
    assert q.output_dir() == (q.root / "output" / "my_set").resolve()
    assert q.choices.get("layout") == "pile" and q.dirty


def test_legacy_session_file_is_a_bare_config(tmp_path):
    p = project_with_brick()
    f = tmp_path / "old_session.yaml"
    p.cfg.save(f)
    q = Project()
    assert q.restore_session(f) and q.cfg.objects.models[0].path == BRICK


def test_checks_report_silent_fallbacks():
    p = Project()
    assert any(i.severity == ERROR and "No part" in i.message for i in p.issues())
    p = project_with_brick()
    assert p.blocking_issues() == []
    p.cfg.scene.ground.texture_mode = "palette"            # no colours -> would silently become procedural
    p.cfg.scene.background.mode = "image_folder"           # no photos  -> would silently become noise
    msgs = [i.message for i in p.issues() if i.severity == WARNING]
    assert any("table colours" in m for m in msgs) and any("photo folder" in m for m in msgs), msgs
    p.cfg.objects.models[0].path = "/does/not/exist.stl"
    assert any("file not found" in i.message for i in p.blocking_issues())


def test_unit_check_flags_absurd_sizes():
    p = project_with_brick()
    p.cfg.objects.models[0].unit = "m"                      # a 50 mm brick read as 50 m
    assert any("check the unit" in i.message for i in p.issues())


def test_dataset_status_detects_mixing(tmp_path):
    p = project_with_brick()
    p.cfg.output.output_dir = str(tmp_path / "ds")
    p.cfg.generation.num_scenes = 4
    st = p.dataset_status()
    assert not st.is_dataset and st.done_scenes == 0 and not st.would_mix
    out = p.output_dir()
    (out / ".progress").mkdir(parents=True)
    p.cfg.save(out / "config.yaml")
    for i in (0, 1):
        (out / ".progress" / f"{i:06d}.done").write_text("{}")
    st = p.dataset_status()
    assert st.is_dataset and st.done_scenes == 2 and st.changed == [] and not st.would_mix and not st.complete
    p.cfg.generation.num_scenes = 2                         # same settings, fewer scenes: complete, no mixing
    assert p.dataset_status().complete and not p.dataset_status().would_mix
    p.cfg.objects.models[0].material.color = [0.1, 0.5, 0.1]   # a setting that changes the images
    st = p.dataset_status()
    assert st.would_mix and st.changed == ["objects.models[0].material.color"]


def test_diff_settings_names_leaves():
    a, b = ProjectConfig().to_dict(), ProjectConfig().to_dict()
    b["sensor"]["tonemap"] = "srgb"
    b["viewpoints"]["radius"]["max"] = 9.0
    assert diff_settings(a, b) == ["sensor.tonemap", "viewpoints.radius.max"]


def test_picked_colour_is_used_for_files_without_colour():
    """An STL has no colours: in 'original' mode the model's colour must be rendered, never a silent grey."""
    assert effective_material_mode("original", asset_has_colors=False) == "fixed_color"
    assert effective_material_mode("original", asset_has_colors=True) == "original"
    m = ObjectModel(path=BRICK).material
    m.color = [0.0, 1.0, 0.0]
    mat = _material(m, np.random.default_rng(0), asset_has_colors=False)
    assert np.allclose(mat["base_color"], (0.0, 1.0, 0.0))
    assert _material(m, np.random.default_rng(0), asset_has_colors=True)["base_color"] is None


def test_class_name_is_the_cad_file_name():
    m = ObjectModel(path="/data/cad/LEGO8-7071-Helmet480-HF.stl")
    assert m.class_name == "LEGO8-7071-Helmet480-HF"                  # nothing typed: the CAD file name
    assert ObjectModel(path=r"D:\\cad\\Gear v2.step").class_name == "Gear v2"     # Windows path, read anywhere
    m.name = "helmet"
    assert m.class_name == "helmet" and m.cad_name == "LEGO8-7071-Helmet480-HF"   # a typed name wins
    m.name = "  "
    assert m.class_name == "LEGO8-7071-Helmet480-HF"


def test_checks_flag_ambiguous_class_names():
    p = project_with_brick()
    p.cfg.objects.models.append(ObjectModel(path=BRICK, name="brick", class_id=7, unit="mm"))
    assert any("different class IDs" in i.message for i in p.issues())
    p.cfg.objects.models[1].class_id = p.cfg.objects.models[0].class_id
    p.cfg.objects.models[1].name = "other"
    msgs = [i.message for i in p.issues()]
    assert not any("different class IDs" in m for m in msgs) and any("share class ID" in m for m in msgs)
