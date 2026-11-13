"""Dataset generation driver (multi-process, deterministic per scene, resumable)."""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .config import ProjectConfig
from .writers import SceneWriter, assign_split, clear_dataset, dataset_stats, export_all, jdump, write_static

class SceneWorker:
    """Renders and writes single scenes. One instance lives in every worker process (it owns that process's
    OpenGL context, loaded meshes and writer)."""

    def __init__(self, cfg: ProjectConfig, out_dir: Path):
        from .scene import SceneSampler
        self.cfg = cfg
        self.sampler = SceneSampler(cfg)
        self.writer = SceneWriter(cfg, Path(out_dir), self.sampler)

    def process(self, idx: int) -> dict:
        """Scene `idx` is a pure function of (config, seed, idx): sample, render every view, write, mark done."""
        cfg, sampler, writer = self.cfg, self.sampler, self.writer
        t0 = time.time()
        try:
            rng = np.random.default_rng([cfg.generation.seed, idx])
            st = sampler.sample_scene(idx, rng)
            want_full = "bop" in cfg.output.formats or cfg.output.save_amodal_masks
            views = []
            for v in range(len(st.views)):
                vrng = np.random.default_rng([cfg.generation.seed, idx, v, 1])
                views.append(sampler.render_view(st, v, vrng, want_normals=cfg.output.save_normals,
                                                 want_full_masks=want_full))
            split = assign_split(idx, cfg.output.split_ratios, cfg.generation.seed)
            stats = writer.write_scene(st, views, split)
            sampler.release_scene_resources()
            m = DatasetGenerator.done_marker(writer.out, idx)
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(json.dumps(stats), encoding="utf-8")
            return {"scene": idx, "ok": True, "stats": stats, "time": time.time() - t0, "info": st.info}
        except Exception as e:  # keep going on single failures
            return {"scene": idx, "ok": False, "error": f"{e}\n{traceback.format_exc()}", "time": time.time() - t0}


_worker: Optional[SceneWorker] = None   # the SceneWorker of this process (set by _init_worker)


def _init_worker(cfg_dict: dict, out_dir: str) -> None:
    import faulthandler
    faulthandler.enable(all_threads=True)   # a native crash in a worker prints its Python stack
    parent = mp.parent_process()
    if parent is not None:      # never outlive the generator (e.g. it was killed by "Cancel"): free GPU + CPU
        threading.Thread(target=lambda: (parent.join(), os._exit(1)), daemon=True).start()
    global _worker
    _worker = SceneWorker(ProjectConfig.model_validate(cfg_dict), Path(out_dir))


def _process_scene(idx: int) -> dict:
    return _worker.process(idx)


class DatasetGenerator:
    """Generates a dataset: deterministic per scene, parallel over processes, resumable."""

    def __init__(self, cfg: ProjectConfig, out_dir: Optional[str] = None, workers: Optional[int] = None,
                 progress: Callable[[dict], None] = print, cancel: Optional[Callable[[], bool]] = None):
        self.cfg = cfg
        self.out = Path(out_dir or cfg.output.output_dir).resolve()
        self.workers = max(1, workers or cfg.generation.workers or (os.cpu_count() or 2) - 1)
        self.progress = progress
        self.cancel = cancel or (lambda: False)

    @staticmethod
    def done_marker(out: Path, idx: int) -> Path:
        return Path(out) / ".progress" / f"{idx:06d}.done"

    def scene_range(self) -> range:
        g = self.cfg.generation
        return range(g.start_scene, g.start_scene + g.num_scenes)

    def pending_scenes(self) -> List[int]:
        return [i for i in self.scene_range() if not self.done_marker(self.out, i).exists()]

    def _log(self, msg: str) -> None:
        self.progress({"event": "log", "msg": msg})

    def run(self, resume: bool = True, allow_mixed: bool = False) -> dict:
        """resume=True keeps finished scenes and renders the rest; resume=False starts the dataset from scratch.
        Resuming a dataset whose finished scenes were made with other settings is refused unless allow_mixed."""
        from .project import scene_relevant_changes
        from .validation import ERROR, check_config
        cfg, out, g = self.cfg, self.out, self.cfg.generation
        issues = check_config(cfg)
        errors = [i.message for i in issues if i.severity == ERROR]
        if errors:
            raise RuntimeError("Cannot generate: " + " | ".join(errors))
        out.mkdir(parents=True, exist_ok=True)
        self.progress({"event": "start", "out": str(out), "scenes": g.num_scenes, "workers": self.workers})
        for i in issues:
            self._log(f"note: {i.message}")
        if not resume and clear_dataset(out):
            # a fresh start must not leave images of an earlier run (other settings, more views ...) in the dataset
            self._log("removed the previous dataset in this folder (fresh start)")
        elif resume and (out / "config.yaml").exists() and len(self.pending_scenes()) < g.num_scenes:
            changed = scene_relevant_changes(ProjectConfig.load(out / "config.yaml").to_dict(), cfg.to_dict())
            if changed and not allow_mixed:
                raise RuntimeError(
                    f"{out} holds finished scenes made with different settings ({', '.join(changed[:8])}"
                    f"{' ...' if len(changed) > 8 else ''}). Start fresh (--no-resume) or, to knowingly keep them, "
                    f"pass --allow-mixed.")
            if changed:
                self._log(f"WARNING: keeping finished scenes made with other settings ({len(changed)} differ): "
                          f"this dataset is a mix of two configurations")

        # warm caches (stable poses / decomposition) once in the main process and write static files
        from .scene import SceneSampler
        t0 = time.time()
        warm = SceneSampler(cfg, renderer=None, log=self._log)
        write_static(cfg, out, warm)
        self._log(f"assets ready in {time.time() - t0:.1f}s")

        todo = self.pending_scenes()
        skipped = g.num_scenes - len(todo)
        if skipped:
            self._log(f"resuming: {skipped} scenes already done")
        self._done, self._failed, self._in_a_row, self._skipped = skipped, 0, 0, skipped
        self._t_start = time.time()
        self._max_in_a_row = max(20, 2 * self.workers)   # that many failures in a row = systematic problem: stop
        cfg_dict = cfg.to_dict()

        if self.workers == 1 or len(todo) <= 1:
            cancelled = self._run_inline(todo, cfg_dict)
        else:
            cancelled = self._run_pool(todo, cfg_dict)

        stopped = self._in_a_row >= self._max_in_a_row
        if stopped:
            self._log(f"STOPPED: {self._in_a_row} scenes in a row failed - this is not random, see the first error "
                      f"above. Fix it and start again (finished scenes are kept).")
        self._log("exporting formats ...")
        summary = export_all(cfg, out, log=self._log)
        stats = dataset_stats(out)
        stats.update(exports=summary, failed_scenes=self._failed, cancelled=cancelled or stopped, stopped=stopped,
                     seconds=round(time.time() - self._t_start, 1))
        jdump(stats, out / "dataset_info.json", indent=2)
        self.progress({"event": "finished", "stats": stats, "out": str(out)})
        return stats

    def _report(self, r: dict) -> None:
        g = self.cfg.generation
        self._done += 1
        if not r["ok"]:
            self._failed += 1
            self._in_a_row += 1
            self.progress({"event": "error", "scene": r["scene"], "msg": r["error"]})
        else:
            self._in_a_row = 0
        rate = (self._done - self._skipped) / max(time.time() - self._t_start, 1e-6)
        eta = (g.num_scenes - self._done) / max(rate, 1e-9)
        self.progress({"event": "progress", "done": self._done, "total": g.num_scenes, "failed": self._failed,
                       "scene": r["scene"], "scene_time": round(r["time"], 2), "eta_s": round(eta, 1),
                       "stats": r.get("stats", {})})

    def _run_inline(self, todo: List[int], cfg_dict: dict) -> bool:
        """Everything in this process (workers = 1). Returns True if cancelled."""
        if todo:
            _init_worker(cfg_dict, str(self.out))
        for i in todo:
            if self.cancel():
                return True
            self._report(_process_scene(i))
            if self._in_a_row >= self._max_in_a_row:
                break
        return False

    def _run_pool(self, todo: List[int], cfg_dict: dict, max_restarts: int = 3) -> bool:
        """Scenes in worker processes. A worker that dies (crash, out of memory) does not hang the run: the pool
        is rebuilt and the unfinished scenes are tried again, a few times. Returns True if cancelled."""
        ctx = mp.get_context("spawn")       # the only start method on Windows; also the safe one with OpenGL
        remaining, restarts = list(todo), 0
        while remaining:
            ex = cf.ProcessPoolExecutor(min(self.workers, len(remaining)), mp_context=ctx,
                                        initializer=_init_worker, initargs=(cfg_dict, str(self.out)))
            futures = {ex.submit(_process_scene, i): i for i in remaining}
            pending, broken, cancelled = set(futures), False, False
            try:
                while pending and not broken:
                    done, pending = cf.wait(pending, timeout=0.5, return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        try:
                            self._report(f.result())
                            remaining.remove(futures[f])
                        except BrokenProcessPool:
                            broken = True
                        except Exception as e:      # e.g. the result could not be sent back
                            self._report({"scene": futures[f], "ok": False, "error": repr(e), "time": 0.0})
                            remaining.remove(futures[f])
                    if self.cancel():
                        cancelled = True
                        break
                    if self._in_a_row >= self._max_in_a_row:
                        return False
            finally:
                # stop handing out scenes; scenes being written right now finish (no half-written files)
                ex.shutdown(wait=not broken, cancel_futures=True)
            if cancelled:
                return True
            if broken:
                restarts += 1
                remaining = [i for i in remaining if not self.done_marker(self.out, i).exists()]
                if restarts > max_restarts:
                    self._log(f"STOPPED: worker processes died {restarts} times (crash or out of memory - see the "
                              f"messages above; try fewer worker processes). {len(remaining)} scenes are not done.")
                    self._in_a_row = self._max_in_a_row
                    return False
                self._log(f"a worker process died (crash or out of memory); restarting the workers "
                          f"({restarts}/{max_restarts}), {len(remaining)} scenes left")
        return False


def generate(cfg: ProjectConfig, out_dir: Optional[str] = None, progress: Callable[[dict], None] = print,
             cancel: Optional[Callable[[], bool]] = None, resume: bool = True, workers: Optional[int] = None,
             allow_mixed: bool = False) -> dict:
    """Function form of DatasetGenerator(...).run(...)."""
    return DatasetGenerator(cfg, out_dir, workers, progress, cancel).run(resume=resume, allow_mixed=allow_mixed)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import faulthandler
    faulthandler.enable(all_threads=True)
    ap = argparse.ArgumentParser(prog="syngt", description="Synthetic GT generator for detectors / pose estimators")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="start the UI (default)")
    g = sub.add_parser("generate", help="generate a dataset from a config")
    g.add_argument("config")
    g.add_argument("--out", default=None)
    g.add_argument("--workers", type=int, default=None)
    g.add_argument("--scenes", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--no-resume", action="store_true", help="start fresh: delete the dataset in the folder first")
    g.add_argument("--allow-mixed", action="store_true",
                   help="resume although the finished scenes were made with other settings")
    g.add_argument("--json-progress", action="store_true", help="machine readable progress on stdout")
    e = sub.add_parser("export", help="re-export COCO/YOLO from native annotations")
    e.add_argument("dataset")
    e.add_argument("--formats", nargs="+", default=None)
    e.add_argument("--min-visib", type=float, default=None)
    e.add_argument("--min-px", type=int, default=None)
    e.add_argument("--max-trunc", type=float, default=None)
    e.add_argument("--obb-source", choices=["visible_mask", "amodal_mask", "projected_3d_box"], default=None)
    p = sub.add_parser("preview", help="render one sample scene to PNGs")
    p.add_argument("config")
    p.add_argument("--scene", type=int, default=0)
    p.add_argument("--out", default="preview")
    sub.add_parser("presets", help="list camera presets")
    lk = sub.add_parser("look", help="camera look profiles (measured from real photos)")
    lks = lk.add_subparsers(dest="look_cmd")
    lc = lks.add_parser("create", help="measure real photos and auto-tune a profile")
    lc.add_argument("folders", nargs="+")
    lc.add_argument("--name", required=True)
    lc.add_argument("--config", required=True, help="project used as the tuning scene")
    lc.add_argument("--out", default=None)
    lc.add_argument("--default", action="store_true", help="use for new projects")
    la = lks.add_parser("apply", help="apply a profile to a project config (in place)")
    la.add_argument("profile")
    la.add_argument("config")
    lks.add_parser("list")
    args = ap.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui.app import run
        return run()
    if args.cmd == "look":
        from .lookprofile import (apply_profile, create_profile, default_profile, list_profiles, load_profile,
                                  report, save_profile, set_default_profile)
        if args.look_cmd == "create":
            prof = create_profile(args.name, args.folders, ProjectConfig.load(args.config))
            path = save_profile(prof, args.out)
            if args.default:
                set_default_profile(path)
            print(report(prof))
            print("saved", path, "(default)" if args.default else "")
        elif args.look_cmd == "apply":
            cfg = ProjectConfig.load(args.config)
            for line in apply_profile(cfg, load_profile(args.profile), args.profile):
                print(line)
            cfg.save(args.config)
        else:
            d = default_profile()
            for p in list_profiles():
                print(("* " if d and p == d else "  ") + p)
        return 0
    if args.cmd == "presets":
        from .cameras import PRESETS, intrinsics_for
        for pr in PRESETS:
            fx, fy, cx, cy = intrinsics_for(pr.color, *pr.color.native)
            print(f"{pr.key:42s} color {pr.color.native} fx~{fx:.0f}  depth {pr.depth.native} {pr.depth_type} B={pr.baseline_m * 1000:.0f}mm")
        return 0
    if args.cmd == "generate":
        cfg = ProjectConfig.load(args.config)
        if args.scenes is not None:
            cfg.generation.num_scenes = args.scenes
        if args.seed is not None:
            cfg.generation.seed = args.seed

        def prog(ev):
            if args.json_progress:
                print(json.dumps(ev), flush=True)
            elif ev["event"] == "progress":
                print(f"[{ev['done']}/{ev['total']}] scene {ev['scene']} {ev['scene_time']}s  eta {ev['eta_s']:.0f}s"
                      + (f"  failed={ev['failed']}" if ev['failed'] else ""), flush=True)
            elif ev["event"] in ("log", "start"):
                print(ev.get("msg", ev), flush=True)
            elif ev["event"] == "error":
                print(f"ERROR scene {ev['scene']}: {ev['msg']}", file=sys.stderr, flush=True)
            elif ev["event"] == "finished":
                print(json.dumps(ev["stats"], indent=2), flush=True)

        try:
            stats = generate(cfg, args.out, prog, resume=not args.no_resume, workers=args.workers,
                             allow_mixed=args.allow_mixed)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            return 2
        return 1 if stats.get("stopped") or stats.get("failed_scenes") else 0
    if args.cmd == "export":
        out = Path(args.dataset)
        cfg = ProjectConfig.load(out / "config.yaml")
        o = cfg.output
        if args.min_visib is not None:
            o.min_visib_fract = args.min_visib
        if args.min_px is not None:
            o.min_px_visib = args.min_px
        if args.max_trunc is not None:
            o.max_truncation = args.max_trunc
        if args.obb_source:
            o.obb_source = args.obb_source
        print(json.dumps(export_all(cfg, out, args.formats), indent=2))
        return 0
    if args.cmd == "preview":
        import cv2
        from .scene import SceneSampler
        from .visualize import draw_overlay
        cfg = ProjectConfig.load(args.config)
        s = SceneSampler(cfg, log=print)
        st = s.sample_scene(args.scene)
        od = Path(args.out)
        od.mkdir(parents=True, exist_ok=True)
        for v in range(len(st.views)):
            for fr in s.render_view(st, v, np.random.default_rng([cfg.generation.seed, args.scene, v, 1])):
                vis = draw_overlay(fr.rgb[..., :3], fr.annotations, s.class_names)
                cv2.imwrite(str(od / f"v{v}_{fr.cam_name}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"wrote preview images to {od.resolve()}")
        return 0
    return 0
