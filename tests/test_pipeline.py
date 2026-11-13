"""End-to-end generation: resume keeps finished scenes, a fresh start leaves nothing of the old dataset behind.

Renders a few tiny images (needs OpenGL; EGL works headless).   Run:  python -m pytest tests/test_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syngt.cameras import camera_from_preset  # noqa: E402
from syngt.config import ALL_FORMATS, IntRange, ObjectModel, ProjectConfig  # noqa: E402
from syngt.pipeline import DatasetGenerator, generate  # noqa: E402


def tiny_cfg(n_scenes=2, views=2) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.cameras = [camera_from_preset("Intel RealSense D435 / D435i", "color", 424, 240, name="cam")]
    cfg.objects.models.append(ObjectModel(path=str(ROOT / "examples" / "models" / "brick_50x10x5.stl"),
                                          unit="mm", count=IntRange(min=2, max=3)))
    cfg.objects.models[0].material.color = [0.0, 0.8, 0.0]
    cfg.viewpoints.views_per_scene = views
    cfg.generation.num_scenes = n_scenes
    cfg.output.formats = list(ALL_FORMATS)
    return cfg


def images(out: Path):
    return sorted(p.name for p in (out / "rgb").rglob("*.jpg"))


def test_generate_resume_and_fresh_start(tmp_path):
    out = tmp_path / "ds"
    events = []
    stats = generate(tiny_cfg(2, views=3), str(out), events.append, resume=True, workers=1)
    assert stats["failed_scenes"] == 0 and len(images(out)) == 6
    assert (out / "yolo" / "data.yaml").exists() and (out / "coco").is_dir() and (out / "bop").is_dir()
    # every export names the class after the CAD file
    assert (out / "yolo" / "classes.txt").read_text().split() == ["brick_50x10x5"]
    assert "0: brick_50x10x5" in (out / "yolo" / "data.yaml").read_text()
    assert json.loads((out / "coco" / "train.json").read_text())["categories"][0]["name"] == "brick_50x10x5"
    assert next(iter(json.loads((out / "bop" / "obj_id_map.json").read_text()).values()))["name"] == "brick_50x10x5"
    ann = json.loads(next((out / "annotations").rglob("*.json")).read_text())
    assert all(o["class_name"] == "brick_50x10x5" for o in ann["objects"])
    first = {p.name: p.stat().st_mtime_ns for p in (out / "rgb").rglob("*.jpg")}

    # resume with more scenes: finished scenes are not rendered again
    gen = DatasetGenerator(tiny_cfg(3, views=3), str(out), workers=1, progress=events.append)
    assert gen.pending_scenes() == [2]
    gen.run(resume=True)
    again = {p.name: p.stat().st_mtime_ns for p in (out / "rgb").rglob("*.jpg")}
    assert len(again) == 9 and all(again[k] == v for k, v in first.items())

    # fresh start with fewer views: no image / annotation of the old run may survive
    stats = generate(tiny_cfg(2, views=1), str(out), events.append, resume=False, workers=1)
    assert len(images(out)) == 2, images(out)
    n_ann = len(list((out / "annotations").rglob("*.json")))
    assert n_ann == 2 and sum(stats["splits"].values()) == 2
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["failed_scenes"] == 0


def test_fresh_start_only_touches_syngt_files(tmp_path):
    out = tmp_path / "ds"
    generate(tiny_cfg(1, views=1), str(out), lambda e: None, resume=True, workers=1)
    (out / "my_notes.txt").write_text("keep me")
    generate(tiny_cfg(1, views=1), str(out), lambda e: None, resume=False, workers=1)
    assert (out / "my_notes.txt").read_text() == "keep me"


def test_resume_refuses_to_mix_settings(tmp_path):
    import pytest
    out = tmp_path / "ds"
    generate(tiny_cfg(1, views=1), str(out), lambda e: None, workers=1)
    changed = tiny_cfg(2, views=1)
    changed.objects.models[0].material.color = [0.9, 0.0, 0.0]          # other images than the finished scene
    with pytest.raises(RuntimeError, match="different settings"):
        generate(changed, str(out), lambda e: None, workers=1)
    log = []
    generate(changed, str(out), log.append, workers=1, allow_mixed=True)   # only when asked for explicitly
    assert any("mix of two configurations" in e.get("msg", "") for e in log)
    more = tiny_cfg(3, views=1)
    more.objects.models[0].material.color = [0.9, 0.0, 0.0]
    more.output.formats = ["native", "bop", "coco"]                        # exported formats can change freely
    generate(more, str(out), lambda e: None, workers=1)
    assert len(images(out)) == 3


def test_export_skips_interrupted_scenes(tmp_path):
    """A scene that was being written when the run was killed must not break or pollute the export."""
    from syngt.writers import export_all, iter_annotations
    out = tmp_path / "ds"
    cfg = tiny_cfg(2, views=1)
    generate(cfg, str(out), lambda e: None, workers=1)
    (out / ".progress" / "000001.done").unlink()                           # scene 1 never finished ...
    victim = next(p for p in (out / "annotations").rglob("000001_*.json"))
    victim.write_text(victim.read_text()[:50])                             # ... and its file is cut off
    assert sum(1 for s in ("train", "val", "test") for _ in iter_annotations(out, s)) == 1
    assert export_all(cfg, out, ["yolo", "coco"], log=lambda s: None)["yolo"]


def test_a_dying_worker_does_not_hang_the_run(tmp_path):
    """imap_unordered waited forever for the scene of a killed worker (crash / out-of-memory kill)."""
    import subprocess
    import textwrap
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(f"""
        import os, signal, sys, threading, time
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(ROOT / 'tests')!r})
        from test_pipeline import tiny_cfg
        from syngt.pipeline import DatasetGenerator
        import multiprocessing as mp

        if __name__ == "__main__":
            killed = []
            def on(ev):
                if ev.get("event") == "progress" and not killed:
                    kids = mp.active_children()
                    if kids:
                        killed.append(kids[0].pid)
                        os.kill(kids[0].pid, signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
                if ev.get("event") == "log":
                    print(ev["msg"], flush=True)
            st = DatasetGenerator(tiny_cfg(8, views=1), {str(tmp_path / 'ds')!r}, workers=2, progress=on).run()
            print("RESULT", st["failed_scenes"], sum(st["splits"].values()), bool(killed), flush=True)
    """))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=240)
    assert "RESULT 0 8 True" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    assert "a worker process died" in r.stdout


def test_cli_exit_code_reports_failure(tmp_path):
    import subprocess
    cfg = tiny_cfg(1, views=1)
    cfg.objects.models[0].path = str(tmp_path / "missing.stl")
    cfg.save(tmp_path / "bad.yaml")
    r = subprocess.run([sys.executable, "-m", "syngt", "generate", str(tmp_path / "bad.yaml"), "--out",
                        str(tmp_path / "o")], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert r.returncode != 0 and "file not found" in r.stderr


def test_coco_bb_export_matches_the_rtdetr_trainer_layout(tmp_path):
    """<dataset>/coco_bb/train/coco_train.json + images, .../validation/coco_validation.json + images
    (what supervision.DetectionDataset.from_coco / the RT-DETR trainer read)."""
    out = tmp_path / "ds"
    cfg = tiny_cfg(6, views=1)
    cfg.output.split_ratios = [0.5, 0.5, 0.0]
    cfg.output.formats = ["native", "coco_bb"]
    generate(cfg, str(out), lambda e: None, workers=1)
    n_img = 0
    for folder, name in (("train", "coco_train.json"), ("validation", "coco_validation.json")):
        d = out / "coco_bb" / folder
        coco = json.loads((d / name).read_text())
        assert coco["categories"] == [{"id": 1, "name": "brick_50x10x5", "supercategory": ""}]
        assert coco["images"] and coco["annotations"]
        for im in coco["images"]:
            assert "/" not in im["file_name"] and (d / im["file_name"]).is_file()     # image sits next to the json
            assert set(im) == {"id", "file_name", "width", "height"}
        ids = {im["id"] for im in coco["images"]}
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            assert a["image_id"] in ids and a["category_id"] == 1 and a["iscrowd"] == 0
            assert w >= 1 and h >= 1 and x >= 0 and y >= 0 and x + w <= 424 and y + h <= 240
            assert abs(a["area"] - w * h) < 0.1
        n_img += len(coco["images"])
    assert n_img == 6
