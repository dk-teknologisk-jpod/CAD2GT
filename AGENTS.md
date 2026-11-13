# Notes for coding agents

## Must run on both Windows and Linux

syngt is used on **Windows PCs** (NVIDIA desktop GPUs and Optimus laptops) **and Linux PCs**, including
NVIDIA Jetson (Orin, aarch64). Every change has to keep working on both. Check these before you finish:

- **Paths:** use `pathlib.Path` and never hard-code `\` or `/` separators or drive letters. Configs store
  paths as plain strings, and they must load on the other OS too.
- **Python:** `requires-python >= 3.10`. Linux/Jetson uses 3.10, Windows uses 3.12, so use no syntax or
  stdlib features newer than 3.10.
- **Multiprocessing:** always use `mp.get_context("spawn")`, the only start method on Windows. Worker code
  must be importable and picklable at module level, and any script that starts workers needs an
  `if __name__ == "__main__":` guard.
- **OpenGL context** (`syngt/renderer.py:_create_context`): EGL first, so Linux renders headless with no
  X display (SSH, locked screen, many workers). Then the platform default (WGL on Windows, GLX on Linux) as
  the fallback. Never call `moderngl.create_standalone_context()` directly anywhere else.
  `SYNGT_GL_BACKEND=egl|default` forces one. GLX in worker processes caused "XOpenDisplay: cannot open
  display" and crashes, so don't reintroduce a dependency on `DISPLAY` in the generation path.
- **Windows-only features** must be guarded with `os.name == "nt"` and hidden on other systems, e.g. the
  Optimus GPU preference in `syngt/gpu.py` (registry / `winreg`) and its Tools-menu entries.
- **Linux-only features** (e.g. `/proc`, `fork`, xcb) must have a Windows path or be guarded the same way.
- **Qt:** PySide6 >= 6.5 on Linux/X11 needs the system package `libxcb-cursor0`. Without it no window
  appears ("Could not load the Qt platform plugin xcb"). Don't rely on other platform-specific Qt plugins.
- **Defaults that depend on the machine** (e.g. worker count) are stored as "auto" (`0`) and resolved at
  runtime (`os.cpu_count()`). Never write a machine-specific value as a default into a config.
- **Launchers:** `run_gui.bat` is for Windows. On Linux use `.venv/bin/python -m syngt gui`. Keep both
  working when changing entry points.
- **User files:** GUI preferences use `QSettings("syngt", "syngt")`, which is the registry on Windows, so
  never derive file paths from `QSettings.fileName()`. Files (session autosave `syngt_session.yaml`,
  `crash.log`) go to `syngt.gui.app.user_dir()`: `%APPDATA%\syngt` on Windows, `~/.config/syngt` on Linux.

When you can, test on the OS you are on, and reason explicitly about the other one: path handling,
process start method, GL backend and anything in `os.name` branches.

## Qt threading rule (caused segfaults)

Workers that run in a `QThread` (`gui/preview.py`, `gui/lookdialog.py`) must talk to widgets only through
signals connected to **bound `@Slot` methods of a GUI object**, never lambdas or plain functions. PySide6 runs
a lambda in the *sender's* thread, so a widget gets changed from the worker thread. That caused random
segfaults in `QWidgetTextControl::setPalette`. Before exit, let a running `QThread` finish
(`PreviewPanel.shutdown`). Destroying it while it runs aborts the process.

## Architecture: backend and UI are separate

```
syngt/                    backend - no Qt anywhere in these modules
  config.py               ProjectConfig (pydantic): every setting, YAML round trip. Pure data.
  project.py              Project: the application model. Owns the config being edited, file path, dirty flag,
                          undo/redo, session save/restore, bulk actions (profiles, looks), dataset status.
  validation.py           check_config(): pre-flight checks -> list of Issue (error / warning / info).
  profiles.py, lookprofile.py, appearance.py, plates.py     bulk actions and measurements used by Project.
  assets.py               MeshAsset: loading, units, stable poses, collision hulls, caches.
  placement.py            physics / placement (MuJoCo).
  scene.py                SceneSampler: samples a scene (parts, materials, ground, lights, views), renders views.
  renderer.py             Renderer (moderngl): rasterisation + GPU sensor model.
  imaging.py, textures.py, geometry.py, annotations.py, viewpoints.py, cameras.py   stateless maths / helpers.
  writers.py              SceneWriter + exporters (native, BOP, COCO, YOLO), clear_dataset().
  pipeline.py             SceneWorker (one per process), DatasetGenerator, CLI (`python -m syngt ...`).
syngt/gui/                PySide6 front end - no rules, no settings of its own
  main_window.py          MainWindow, GeneratePanel: show a Project, forward actions, react to its events.
  widgets.py              Binder (editors bound to config fields), WheelGuard.
  quick.py, pages.py      Quick setup and the detailed tabs.
  preview.py, browser.py, realism.py, lookdialog.py, rectify.py
  icon.py                 the app icon, drawn in code (assets/syngt.png + .ico are exports for shortcuts:
                          regenerate them with icon.write_icon_files after changing the drawing).
```

Rules that keep it that way:

- **The backend never imports `syngt.gui`** (except the CLI's lazy `gui` command), and never Qt.
- **Every change to settings goes through the `Project`**: `project.set(obj, attr, value)` for one field,
  `project.touch(label)` after an in-place edit, `with project.batch(label):` / `project.apply(label, fn)` for bulk
  actions. In the GUI use `Binder` (`b.dspin(...)`, `b.combo(...)`, `b.flags(...)`, `b.path(...)`, `b.color(...)`).
  Never write `cfg.x = y` from a signal handler without it: the dirty flag, undo, auto-save, preview and the
  checks panel would miss the change.
- **Events:** `EDIT` = one field changed in place (bound widgets stay valid). `BULK` = many settings changed and
  sub-objects may have been *replaced* (e.g. `apply_profile` replaces `cfg.sensor`), so the visible page is rebuilt.
  `RESET` = `project.cfg` is a new object (new / open / undo), so all pages are rebuilt. Pages are cheap, rebuildable
  views: they read `project.cfg` when built and keep no state of their own. Anything a page must remember across
  a rebuild belongs in the `Project` (`project.choices`).
- **No silent fallbacks.** If the generator would fall back to something else or ignore a setting in the current
  mode, add a check to `validation.check_config()` so it shows in the checks panel above *Generate*. In the GUI
  show only the rows that have an effect in the current mode (`pages.show_rows`).
- **Bulk actions report what they changed.** `Project.apply()` returns the names of the changed settings; the
  window shows them (`MainWindow.report_changes`). A profile must only touch the settings it is about.
- **Never mix datasets.** `Project.dataset_status()` detects finished scenes made with other settings; the GUI
  asks before generating. `resume=False` clears the old dataset first (`writers.clear_dataset`).
- **The mouse wheel never edits.** `WheelGuard` (installed in `gui/app.py`) redirects wheel events on combo
  boxes, spin boxes and sliders to the page. Use plain Qt widgets; do not add wheel handling per widget.
- **Export formats** are listed once in `config.ALL_FORMATS` (and `DETECTOR_FORMATS` for those rebuilt by
  re-export). A new format needs: an exporter in `writers.py` called from `export_all`, its folder in
  `writers.DATASET_ENTRIES`, labels in `gui/pages.FORMAT_HELP` and `gui/quick.FORMAT_LABELS`, and a layout test.
  `coco_bb` is the layout of the user's RT-DETR trainer (`supervision.DetectionDataset.from_coco`).
- **Class names are the CAD file names.** `ObjectModel.class_name` (the file name without extension, unless
  the user typed another name) is the only source for names in labels and exports (native, BOP, COCO, YOLO
  `data.yaml` / `classes.txt`). Never derive a class name any other way.
- **Material colour:** mode `original` uses the file's texture/colours; files without any (STL, STEP) are rendered
  in the model's `color` (`scene.effective_material_mode`). Picking a colour in the UI selects `fixed_color`.
- **Exposure:** auto exposure meters the scene light by default (`sensor.ae_metering = "light"`): an 18 % grey
  card is shown at `ae_target`, and the brightest part is kept below `scene.HIGHLIGHT_LIMIT`
  (`scene.light_metered_exposure`). Part brightness must not depend on the table or background. Look profiles
  store their metering mode; older ones are converted on apply (`lookprofile.light_metering_target`).
- **Generation must not hang or lie:** workers run in a `ProcessPoolExecutor` (a dead worker rebuilds the pool),
  workers exit when the generator dies, JSON is written atomically, exports skip scenes without a `.done` marker,
  and the CLI returns non-zero when scenes failed. Resuming a dataset made with other settings is refused unless
  `--allow-mixed` is given.
- **Known limits (reported by the checks panel, not fixed):** BOP export cannot express scale jitter or lens
  distortion. Collision hulls are simplified to 64 vertices, so very round parts can sink up to ~0.4 mm.
- Prefer classes for things with state or a life cycle (Project, SceneSampler, Renderer, DatasetGenerator,
  SceneWorker, SceneWriter) and plain functions for stateless maths (imaging, geometry, annotations).

## Layout and checks

- Package `syngt/`: `config.py` (pydantic settings and defaults), `scene.py` (scene sampling + render
  driver), `placement.py` (MuJoCo physics), `renderer.py` (moderngl), `pipeline.py` (CLI + worker pool),
  `writers.py` (output formats), `gui/` (PySide6).
- Tests (`pip install pytest` in the venv; it is not in `requirements.txt`):
  - `python -m pytest tests/test_project.py` - application model, checks, undo, session (no Qt, no rendering).
  - `QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_flow.py` - the window as a user drives it: wheel
    guard, colour picking, profiles + undo, stale-widget cases, checks panel.
  - `python -m pytest tests/test_pipeline.py` - end-to-end generation, resume, fresh start, dead worker, exit codes.
  - `python -m pytest tests/test_regressions.py` - one test per bug that was fixed (physics, assets, viewpoints,
    annotations, exports, look profiles).
  - `python tests/test_consistency.py` and `python tests/test_gpu_postprocess.py` - GT vs. render consistency.
  Run all of them after any change; add a test for every bug you fix.
- Headless generation check (Linux): `env -u DISPLAY .venv/bin/python -m syngt generate <cfg> --out <dir>`.
