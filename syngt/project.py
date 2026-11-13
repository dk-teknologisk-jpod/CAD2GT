"""Application model: the project being edited, independent of any user interface.

The GUI (and any other front end) talks to a :class:`Project` instead of poking at the configuration directly:

* every change goes through :meth:`Project.set`, :meth:`Project.touch` or :meth:`Project.apply`, so the dirty flag,
  undo history and all listeners always see it - there is no way to change a setting "behind the back" of the app;
* bulk actions (profiles, measured looks ...) report exactly which settings they changed;
* :meth:`Project.issues` lists everything that would silently fall back or be ignored;
* :meth:`Project.dataset_status` tells whether the output folder already holds scenes made with other settings.

No Qt in here. Listeners are plain callables.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import yaml

from .config import ProjectConfig
from .validation import ERROR, Issue, check_config

REPO_ROOT = Path(__file__).resolve().parents[1]

EDIT, BULK, RESET = "edit", "bulk", "reset"
# EDIT : one field changed in place (widgets bound to it stay valid)
# BULK : many settings changed at once, sub-objects may have been replaced (views must re-read the config)
# RESET: project.cfg is a new object (new / open / undo)

# settings that do not change what a finished scene looks like (so a dataset can be extended / re-exported
# with different values without mixing two kinds of images)
_NOT_SCENE_RELEVANT = ("name", "look_profile", "generation.num_scenes", "generation.workers",
                       "generation.start_scene", "output.output_dir", "output.formats", "output.min_visib_fract",
                       "output.min_px_visib", "output.max_truncation", "output.obb_source")


def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    """Nested dict/list -> {'a.b[2].c': value} (leaf values only)."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list) and any(isinstance(x, (dict, list)) for x in d):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def diff_settings(a: dict, b: dict) -> List[str]:
    """Dotted names of the settings that differ between two config dicts."""
    fa, fb = flatten(a), flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, None) != fb.get(k, None))


def scene_relevant_changes(old: dict, new: dict) -> List[str]:
    """Settings that differ between two config dicts AND change what a generated scene contains."""
    changed = [k for k in diff_settings(old, new) if not k.startswith(_NOT_SCENE_RELEVANT)]
    # BOP files are written while a scene is rendered (unlike COCO / YOLO, which are exported afterwards)
    if ("bop" in old.get("output", {}).get("formats", [])) != ("bop" in new.get("output", {}).get("formats", [])):
        changed.append("output.formats (BOP)")
    return changed


@dataclass
class DatasetStatus:
    out: Path
    is_dataset: bool = False            # folder was written by syngt (has config.yaml)
    foreign_files: bool = False         # folder is not empty and not a syngt dataset
    done_scenes: int = 0                # finished scenes within the range that would be generated now
    total_scenes: int = 0
    changed: List[str] = field(default_factory=list)   # settings that differ from those of the finished scenes

    @property
    def would_mix(self) -> bool:
        """Continuing would put images made with two different sets of settings into one dataset."""
        return self.done_scenes > 0 and bool(self.changed)

    @property
    def complete(self) -> bool:
        return self.total_scenes > 0 and self.done_scenes >= self.total_scenes


class Project:
    HISTORY = 60

    def __init__(self, root: Path = REPO_ROOT):
        self.root = Path(root)
        self.cfg = ProjectConfig()
        self.path: Optional[Path] = None
        self.dirty = False
        self.revision = 0                       # bumped on every change; views compare it to know they are stale
        self.choices: Dict[str, str] = {}       # last picked quick-setup profiles (shown in the UI only)
        self._listeners: List[Callable[[str], None]] = []
        self._undo: List[Tuple[str, dict]] = []
        self._redo: List[Tuple[str, dict]] = []
        self._committed: dict = self.cfg.to_dict()
        self._last_label, self._last_time = "", 0.0
        self._batch = 0

    # ------------------------------------------------------------------ observers
    def subscribe(self, cb: Callable[[str], None]) -> Callable[[], None]:
        """Call cb(kind) after every change (kind: EDIT, BULK or RESET). Returns the unsubscribe function."""
        self._listeners.append(cb)

        def unsubscribe():
            if cb in self._listeners:
                self._listeners.remove(cb)
        return unsubscribe

    def _emit(self, kind: str) -> None:
        self.revision += 1
        for cb in list(self._listeners):
            cb(kind)

    # ------------------------------------------------------------------ editing
    def set(self, obj: Any, attr: str, value: Any, label: str = "") -> bool:
        """Set one field of a config object. Returns False (and does nothing) if the value is unchanged."""
        if getattr(obj, attr) == value:
            return False
        setattr(obj, attr, value)
        return self.touch(label or attr)

    def touch(self, label: str = "edit", kind: str = EDIT) -> bool:
        """Record a change that was made in place (list edited, several fields set ...)."""
        if self._batch:
            return True
        new = self.cfg.to_dict()
        if new == self._committed:
            if kind != EDIT:
                # same values, but a bulk action may have replaced sub-objects (e.g. cfg.sensor): views that hold
                # references to the old ones must still re-read the configuration
                self._emit(kind)
            return False
        now = time.monotonic()
        # one undo step per "gesture": dragging a slider or typing in one field does not flood the history
        if kind != EDIT or label != self._last_label or now - self._last_time > 1.0:
            self._undo.append((label, self._committed))
            del self._undo[:-self.HISTORY]
        self._redo.clear()
        self._last_label, self._last_time = label, now
        self._committed = new
        self.dirty = True
        self._emit(kind)
        return True

    @contextmanager
    def batch(self, label: str) -> Iterator[None]:
        """Group several changes into one notification / undo step."""
        self._batch += 1
        try:
            yield
        finally:
            self._batch -= 1
        if not self._batch:
            self.touch(label, BULK)

    def apply(self, label: str, fn: Callable[[ProjectConfig], Any]) -> List[str]:
        """Run a bulk action (profile, measured look ...) and return the names of the settings it changed."""
        before = self._committed
        with self.batch(label):
            fn(self.cfg)
        return diff_settings(before, self._committed)

    # ------------------------------------------------------------------ undo / redo
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def _restore(self, state: dict) -> None:
        self.cfg = ProjectConfig.model_validate(state)
        self._committed = self.cfg.to_dict()
        self._last_label = ""
        self.dirty = True
        self._emit(RESET)

    def undo(self) -> Optional[str]:
        """Take back the last change. Returns its label (None if there is nothing to undo)."""
        if not self._undo:
            return None
        label, state = self._undo.pop()
        self._redo.append((label, self._committed))
        self._restore(state)
        return label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        label, state = self._redo.pop()
        self._undo.append((label, self._committed))
        self._restore(state)
        return label

    # ------------------------------------------------------------------ files
    def _replace(self, cfg: ProjectConfig, path: Optional[Path], dirty: bool) -> None:
        try:    # one-off correction of projects saved with a mis-read look-profile brightness target
            from .lookprofile import fix_legacy_look
            dirty = fix_legacy_look(cfg) or dirty
        except Exception:
            pass
        self.cfg, self.path, self.dirty = cfg, path, dirty
        self._committed = cfg.to_dict()
        self._undo.clear()
        self._redo.clear()
        self._last_label = ""
        self._emit(RESET)

    def new(self) -> None:
        cfg = ProjectConfig()
        try:   # new projects start with the default camera look, if one is set
            from .lookprofile import apply_profile, default_profile, load_profile
            p = default_profile()
            if p:
                apply_profile(cfg, load_profile(p), p)
        except Exception:
            pass
        self.choices = {}
        self._replace(cfg, None, False)

    def load(self, path: str | Path) -> None:
        cfg = ProjectConfig.load(path)          # raises on a broken file; the current project stays untouched
        self.choices = {}
        self._replace(cfg, Path(path).resolve(), False)

    def save(self, path: Optional[str | Path] = None) -> Path:
        if path is not None:
            self.path = Path(path).resolve()
        if self.path is None:
            raise ValueError("the project has no file name yet")
        self.cfg.save(self.path)
        self.dirty = False
        return self.path

    def output_dir(self) -> Path:
        """Absolute dataset folder (relative paths are relative to the project file, else the app folder)."""
        p = Path(self.cfg.output.output_dir or "output/dataset")
        if not p.is_absolute():
            p = ((self.path.parent if self.path else self.root) / p).resolve()
        return p

    # ------------------------------------------------------------------ session (continue where you left off)
    def save_session(self, file: Path) -> None:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        data = {"session": 1, "path": str(self.path) if self.path else "", "dirty": bool(self.dirty),
                "choices": dict(self.choices), "config": self.cfg.to_dict()}
        tmp = file.with_name(file.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        tmp.replace(file)

    def restore_session(self, file: Path) -> bool:
        """Load the state stored by save_session (paths are taken as they are - never re-based)."""
        file = Path(file)
        if not file.exists():
            return False
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "config" not in data:                # first session format: the bare config
            data = {"config": data, "path": "", "dirty": True, "choices": {}}
        cfg = ProjectConfig.model_validate(data["config"])
        p = Path(data["path"]) if data.get("path") else None
        self.choices = dict(data.get("choices") or {})
        self._replace(cfg, p, bool(data.get("dirty", False)) or p is None)
        return True

    # ------------------------------------------------------------------ bulk actions (each returns what changed)
    def apply_layout_profile(self, key: str) -> List[str]:
        from .profiles import LAYOUT_PROFILES, apply_layout_profile
        self.choices["layout"] = key
        return self.apply(f"Layout: {LAYOUT_PROFILES[key].title}", lambda c: apply_layout_profile(c, key))

    def apply_view_profile(self, key: str) -> List[str]:
        from .profiles import VIEW_PROFILES, apply_view_profile
        self.choices["view"] = key
        return self.apply(f"Camera position: {VIEW_PROFILES[key].title}", lambda c: apply_view_profile(c, key))

    def apply_background_profile(self, key: str) -> List[str]:
        from .profiles import BACKGROUND_PROFILES, apply_background_profile
        return self.apply(f"Background: {BACKGROUND_PROFILES[key].title}", lambda c: apply_background_profile(c, key))

    def apply_look(self, profile_path: str, lighting: bool = True) -> List[str]:
        from .lookprofile import apply_profile, load_profile
        prof = load_profile(profile_path)
        return self.apply(f"Camera look: {prof.get('name', Path(profile_path).stem)}",
                          lambda c: apply_profile(c, prof, profile_path, lighting=lighting))

    def randomise_look(self) -> List[str]:
        from .profiles import randomise_camera
        return self.apply("Camera look: randomised", randomise_camera)

    # ------------------------------------------------------------------ checks
    def issues(self) -> List[Issue]:
        from .assets import model_extent
        return check_config(self.cfg, model_extent)

    def blocking_issues(self) -> List[Issue]:
        return [i for i in self.issues() if i.severity == ERROR]

    # ------------------------------------------------------------------ dataset on disk
    def dataset_status(self) -> DatasetStatus:
        out = self.output_dir()
        g = self.cfg.generation
        st = DatasetStatus(out=out, total_scenes=g.num_scenes)
        if not out.exists():
            return st
        old_cfg = out / "config.yaml"
        st.is_dataset = old_cfg.exists()
        st.foreign_files = not st.is_dataset and any(p.name != ".syngt_run_config.yaml" for p in out.iterdir())
        prog = out / ".progress"
        if prog.is_dir():
            st.done_scenes = sum(1 for i in range(g.start_scene, g.start_scene + g.num_scenes)
                                 if (prog / f"{i:06d}.done").exists())
        if st.is_dataset and st.done_scenes:
            try:
                with open(old_cfg, "r", encoding="utf-8") as f:
                    old = ProjectConfig.model_validate(yaml.safe_load(f) or {}).to_dict()
                st.changed = scene_relevant_changes(old, self.cfg.to_dict())
            except Exception:
                st.changed = ["(the old dataset's config.yaml cannot be read)"]
        return st

    def write_run_config(self) -> Tuple[Path, Path]:
        """Freeze the current settings for a generation run. Returns (config file, output folder)."""
        out = self.output_dir()
        out.mkdir(parents=True, exist_ok=True)
        run_cfg = self.cfg.copy_deep()
        run_cfg.output.output_dir = str(out)
        p = out / ".syngt_run_config.yaml"
        run_cfg.save(p)
        return p, out
