# syngt – synthetic ground truth from CAD

A desktop app plus CLI that turns CAD models into training data for 2D/OBB/instance-segmentation detectors and
6D pose estimators. You can use one camera or several (multi-view), with physically plausible placements
and real camera models (Intel RealSense, Orbbec, or your own calibration).

Runs on **Windows and Linux**, including NVIDIA Jetson.

```
run_gui.bat                          # UI (Windows)
.venv\Scripts\python -m syngt generate examples\example_config.yaml

.venv/bin/python -m syngt gui        # UI (Linux)
.venv/bin/python -m syngt generate examples/example_config.yaml
```

## Install

**Linux** (including Jetson):

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
sudo apt install libxcb-cursor0      # needed by Qt >= 6.5 for the GUI window on X11
```

Rendering uses EGL, so it needs no display. Generation also works over SSH or with the screen locked.

**Windows:**

The venv lives in `.venv`, built from `D:\workspace\Python3.12`. To recreate it:

```
D:\workspace\Python3.12\python.exe -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

**GPU:** rendering *and* the whole sensor model run on the GPU through OpenGL 3.3 (moderngl, headless):
* rasterisation, shadows, MSAA, ID/depth/normal buffers
* lens-distortion remap, background compositing, exposure, white balance, vignetting, defocus and motion
  blur, shot/read noise, tone mapping

Only the final 8-bit image and the GT buffers are read back. CUDA is not used: it has no rasteriser, and
nvdiffrast would need PyTorch, the CUDA toolkit and an MSVC build.

On Optimus laptops, Windows gives OpenGL to the Intel iGPU unless the executable is marked
*High performance*. This repo's interpreters (`.venv\Scripts\python(w).exe` and the base
`D:\workspace\Python3.12\python(w).exe`) are already marked that way. If you move the venv, use
*Tools → Use high-performance (NVIDIA) GPU for Python*, or Windows Settings → Display → Graphics.
The preview status line shows which GPU renders.

Measured on an RTX 3500 Ada, per 4K frame (render + sensor + full GT):

| setup | time |
|---|---|
| Intel iGPU, CPU sensor model | 4.2 s |
| NVIDIA, GPU sensor model | 0.5 s |

At 720p with 2 cameras and 4 workers, the example produces about 9 images/s, including writing all formats.
`tests/test_gpu_postprocess.py` checks that the GPU sensor model matches the CPU reference
(*Sensor & Render → GPU* toggle) to within ±1 grey level.

## Defaults (6D pose of small parts)

Out of the box, settings target pose estimation of small parts (e.g. 50 × 10 × 5 mm bricks) on a table.
The example project (`examples/example_config.yaml`) is exactly that brick.

| area | defaults |
|---|---|
| Camera | RealSense D435 colour 1280×720, stereo depth model |
| Viewpoints | Fibonacci coverage, 8 views per scene, **absolute distance 0.30–0.60 m** (realistic object pixel scale; D435 min-Z ≈ 0.28 m), elevation 25–90°, roll ±20°, look-at a random object |
| Placement | physics drop on a 25 × 20 cm area, cluster radius 1.5–8 cm (piles / leaning / scattered), 2–5 instances per model, 1 ms physics step, primitive distractors on |
| Output | **native + BOP only**; switch on COCO/YOLO variants when you need detector data |
| Size | 250 scenes × 8 views = 2000 images per camera, workers = auto (all CPU cores but one) |
| Symmetry | cuboid parts: set *box (180° about x, y, z)* so BOP `models_info.json` carries the 3 discrete symmetries |

## Camera look profiles (measured from real photos)

A look profile captures how **your camera** renders the world, measured once from real images:
* brightness and its spread, contrast, colour cast (white-balance bias), noise and sharpness,
* the lighting the exposure was tuned with (optional when applying: *incl. lighting*).

**Auto exposure** (on by default, like the real camera): every image is exposed so that its mean brightness hits
`ae_target`. A profile sets this target to the real brightness range (p10–p90), so any lighting, bright white
parts or dark tables give correctly exposed images. The exposure range is only used with auto exposure off, and
for parts composited onto real photos.

A profile is **camera only**: backgrounds and tables are never changed by it. The background is a separate
option (Quick setup → *3 · Background*): real-world photos / clean plates, or **real table colours**
measured from photos (also on the Scene & Light tab). That measures the work-surface colours of real photos, masking annotated objects, and
uses them as ground colours.

Creating a profile is a closed loop: synthetic frames are rendered, measured and the sensor settings are
adjusted until they match the real statistics. Exposure uses a secant step, and the best iteration is kept.
Saturation and contrast stay within a plausible camera range (±30 %), because a scene's colourfulness comes
from its content, not from the camera.

* **Default:** `looks/orbbec_gemini335.json`, built from 136 Gemini 335 images (`annotation_tool/lego_images`,
  `lego_images-shield`, `made_images`). New projects get it automatically (`looks/DEFAULT`).
* **UI:** Quick setup → *4 · Camera colour & lighting*. Pick a profile, *Extract from real photos…*,
  *Set as default*, or *Randomised camera & lighting*.
* **CLI:**
  * `python -m syngt look create <folders…> --name "my cam" --config project.yaml --default`
  * `python -m syngt look apply looks/x.json project.yaml`
  * `python -m syngt look list`
* **Background photos** (image-folder backgrounds, e.g. clean plates) already carry the real camera's look.
  They are composited **after** the simulated sensor, so only the rendered parts and their shadows get
  exposure, noise, blur and the tone curve.

## Matching your real camera images (sim-to-real)

1. **Quick setup → 3 · Background → Real-world background (photos).** Parts are rendered onto photos of your
   real, **empty** workspace (*Photo folder*), with an invisible table that only catches the soft shadows.
   Alternatives: *Real table colours (measured)*, *Neutral tables*, *Random colourful*.
   Step *4 · Camera colour & lighting* is independent of the background.
2. **No empty photos yet?** Use *Build clean plates from annotated real photos…*. It groups your labelled photos
   by camera pose, masks the labelled parts (YOLO box / polygon `.txt` or COCO json) and takes the per-pixel
   median. It needs ≥5 photos per pose with moved parts; always review the plates. Simply photographing the
   empty table 5–20 times works even better.
3. **Colours:** set the material to *colour palette* (e.g. *LEGO greys/white/black*) instead of random colours,
   and keep distractor saturation low.
4. **Match camera look to real photos…** measures brightness, contrast, saturation, colour cast, noise and
   sharpness of your real images and of 8 synthetic frames, then adjusts exposure, noise, white balance,
   saturation, contrast and blur.
5. Match **distance and density** to the real images using the Quick setup readout. Compare px per part with
   the real photos.

`examples/lego_realistic.yaml` is a worked example for a Gemini 335 over the real workcell, using plates in
`backgrounds/workcell_plates`. Limits: the background photo is 2D, so it fits best for viewpoints close to the
photos' (top view). Parts can land on objects that are only in the photo (e.g. a fixture), because physics
only knows the flat table.

## Workflow (UI)

0. **★ Quick setup** is the whole workflow on one page, top to bottom:
   1. *Parts:* add CAD files; per part its colour and how many per scene.
   2. *How the parts lie* (scattered, pile, separated, bin, floating) and 3. *where the camera is* (full variety,
      top view, 45° overview, robot wrist, low/side, fixed, turntable).
   4. *Background* and 5. *camera colour & lighting* (a measured look profile, or randomised).
   6. *Fine-tune* with sliders (distance in cm, viewing angle, roll, spread, distractors, views per scene). A live
      read-out shows the field of view on the table and how many pixels each part spans.
   7. *Dataset:* output folder, number of scenes, label formats. Then press **Generate**.

   The preview re-renders automatically (mouse wheel zooms it, drag pans, double-click fits).

   What keeps it predictable:
   * **Checks above the Generate button** list everything that would not work as set up (missing files, an empty
     photo folder that would fall back to noise, a wrong unit …). Double-click an entry to open the tab that fixes it.
   * **A profile tells you what it changed** (status bar), and **Ctrl+Z / Ctrl+Shift+Z** undo and redo any change.
   * **The mouse wheel only scrolls.** It never changes a drop-down, number or slider under the pointer.
   * **Only settings that have an effect are shown** for the chosen mode.
   * **A picked colour is always used.** Files without their own colour (STL, STEP) are shown in the part's colour.
   * **Datasets are never mixed.** If the output folder holds scenes made with other settings, Generate asks whether
     to start fresh or keep them.
   * The app **continues where you left off** (unsaved edits included).
1. **Objects:** add CAD files (STL, OBJ, PLY, GLB/GLTF, OFF, 3MF, DAE, STEP via `cascadio`). For each one, set the
   unit (auto-guess shown by *Inspect mesh*), the class ID, the instance count, the colour (from the file, one
   colour with HSV variation, a palette, or random colours) and the roughness/metallic ranges. You can
   also set a symmetry, which is written to BOP `models_info.json`.
   **The CAD frame is kept**, so the GT pose refers to your CAD coordinate system.
   Distractors are primitives and/or a folder of meshes. They are rendered, they occlude and they collide,
   but they are never labelled.
2. **Placement & Physics**
   * *Free floating:* random 6D poses in a volume (uniform SO(3), stable pose or upright), with optional overlap
     avoidance.
   * *Stable poses, separated:* quasi-static stable poses from hull + COM, weighted by probability. Random yaw
     and optional tilt, non-overlapping footprints with min spacing, then an optional short physics settle.
   * *Physics drop:* MuJoCo. The **cluster radius** controls the result: a small radius gives piles where parts
     lean on and occlude each other, a large radius gives scattered parts at rest. You can also drop
     sequentially (stacks) and pick the initial orientation, friction, damping and timestep.
   * *Physics bin:* bin picking with rendered bin walls.
   * Collision geometry is the convex hull, or CoACD convex decomposition for concave parts (brackets, tubes).
3. **Scene & Lighting:** ground plane (solid, procedural, image folder, or **rectified phone photos**),
   backdrop (solid, gradient, noise, image folder, or transparent RGBA), and 0–8 point/directional lights with
   irradiance, colour temperature, elevation and distance ranges, hemisphere ambient and shadows.
4. **Cameras:** presets for RealSense D405/D415/D435(i)/D455/D457/D555/L515, Orbbec Gemini 2/2L/2XL/335/335L/336,
   Astra 2, Femto Bolt/Mega (NFOV/WFOV) and Azure Kinect, plus generic industrial cameras. Pick the stream
   (colour/depth) and the vendor resolution mode; intrinsics are scaled/cropped the way the vendors do it.
   Preset values are **nominal (datasheet FOV)**. For exact results, import your unit's calibration:
   * rs-enumerate-devices / pyrealsense JSON
   * OpenCV YAML/XML
   * ROS `camera_info`
   * Kalibr camchain (builds a multi-camera rig)
   * plain JSON

   Full Brown-Conrady distortion (k1..k6, p1, p2) is rendered exactly: the image is rendered on an
   undistorted, padded canvas and remapped. Per camera you also get intrinsics jitter, rig extrinsics
   (T_rig_cam), a fixed world pose, and a depth sensor model:
   * stereo: σz = z²·σd/(f·B), edge occlusion bands, holes
   * ToF: σ = a·z + b, flying pixels
   * ideal
5. **Viewpoints:**
   * Sampling modes: random on a sphere cap, Fibonacci (even coverage), exhaustive azimuth × elevation grid,
     orbit/turntable, or fixed calibrated poses with jitter.
   * Distance is either absolute or a *fill fraction*: the scene's bounding sphere relative to the half FOV,
     where > 1 deliberately creates truncation.
   * Rigid rig (cameras move together) or independent cameras.
   * Look-at target with jitter, roll, and a minimum number of objects in view.
6. **Sensor & Render:** exposure, white balance, vignetting, filmic/sRGB tone mapping, defocus, motion blur,
   shot and read noise, saturation/contrast and JPEG artefacts. Renderer settings are MSAA, clip planes and
   canvas padding (needed for amodal boxes and truncation).
7. **Output & Generation:** choose formats, label filters (min visibility, min pixels, max truncation), the
   OBB source, keypoints and the train/val/test split (by scene, so there is no leakage between splits),
   then **Generate**.
   * Generation runs in a separate process with N workers and live progress/ETA, and can be cancelled and
     resumed.
   * *Re-export formats only* rebuilds COCO/YOLO with new filters without re-rendering.
   * *Browse dataset* shows every image with GT overlays, including all cameras of a multi-view frame side by
     side.

**Preview (F5):** renders a full sample with the current settings. You can step through views and cameras,
switch layers (RGB, sensor depth, GT depth, instances, semantic, normals) and toggle overlays (masks, bbox,
amodal bbox, OBB, 3D box, pose axes, keypoints, filtered annotations).

### Phone photo → ground texture (Tools menu, or the button on the Scene page)

Photograph your real table or conveyor with a known rectangle lying on it (an A4 sheet works). Click its 4
corners and enter its size. The tool then:

* rectifies the photo to a fronto-parallel texture,
* optionally removes low-frequency shading (so it behaves like albedo),
* optionally makes it tileable,
* saves it with a `.json` sidecar holding its metric size.

The ground plane then tiles it **at true scale**, under every sampled viewpoint and with shadows. Photos of
the real workcell can also go into the *Background image folder* to be used as backdrops.

## Determinism and scaling

Scene *i* is fully determined by `(seed, i)`. To extend a dataset, raise `num_scenes` or set
`start_scene`. Already-finished scenes are skipped (resume); `--no-resume` (in the UI: *Start fresh*) deletes the old dataset in the folder first, so runs never mix. The config is copied into every
dataset as `config.yaml`.

## Output layout (`output_dir`)

| path | content |
|---|---|
| `rgb/<split>/<scene>_<view>_<cam>.jpg` | images (RGBA PNG if the background is transparent) |
| `depth/…png` | uint16 depth (aligned to that camera), value × `depth_scale_mm` = mm, sensor noise model applied |
| `masks/…png` | instance ids (= `inst_id`, 0 = background) as an **indexed/palette PNG**: viewers show colours, the pixel values are the ids. Read with `np.array(PIL.Image.open(p))` or `syngt.writers.read_label_png` (cv2 would convert it to colour). >255 instances fall back to uint16 |
| `semantic/`, `normals/`, `amodal/` | optional |
| `annotations/<split>/<stem>.json` | **native GT** (source of truth), see below |
| `frames/<split>/<scene>_<view>.json` | multi-view group: all cameras of one frame with T_world_cam |
| `cameras.json` | intrinsics, distortion, T_rig_cam, depth models, conventions |
| `bop/` | BOP: `<split>_pbr/<scene>/rgb, depth, mask, mask_visib, scene_gt(_info), scene_camera`; `models/obj_*.ply` (mm) + `models_info.json` (with symmetries). With several cameras, files get a `_<cam>` suffix (BOP-H3 style) |
| `coco/<split>.json` | instances (polygons), plus `bbox_amodal`, `obb`, 3D-box keypoints, `pose` {R, t_mm}, visib/truncation, and per-image K/dist/T_world_cam |
| `coco_bb/` | Plain COCO boxes, one self-contained folder per split: `train/coco_train.json` and `validation/coco_validation.json`, each next to its images (hard links). Read by `supervision.DetectionDataset.from_coco`, e.g. the RT-DETR trainer: point it at `<dataset>/coco_bb` |
| `yolo/`, `yolo_obb/`, `yolo_seg/`, `yolo_pose/` | Ultralytics layout, `data.yaml`; images are hard links (no extra disk use). The class index is the dense index of the sorted class_id |

Native per-object fields:
* identity: `inst_id`, `obj_id` (class_id), `class_name`, `model`, `scale`
* pose: `T_cam_obj` / `T_world_obj` (m), `cam_R_m2c` / `cam_t_m2c_mm`
* boxes: `bbox_obj` (amodal, may extend beyond the image), `bbox_visib`
* visibility: `px_count_all` / `px_count_visib`, `visib_fract`, `truncation`
* oriented boxes: `obb_visib`, `obb_amodal`, `obb_box3d` (cx, cy, w, h, angle, corners)
* keypoints: centre + 8 corners with visibility (0 = outside, 1 = occluded, 2 = visible)
* projections: `bbox3d_corners_px`, `origin_px`
* masks: `segmentation_polygons`, `rle_visib`, `rle_amodal` (COCO RLE)

YOLO-pose: 3D-box keypoints are not mirror-symmetric, so train with `fliplr=0`.

## Conventions

* World is z-up in metres, with the support plane at z = 0. Cameras use the OpenCV convention
  (x right, y down, z forward).
* `T_a_b` maps points from frame b to frame a.
* BOP files are in mm.
* Depth is z-depth, not ray length.

## CLI

```
python -m syngt gui
python -m syngt generate cfg.yaml [--out DIR] [--workers N] [--scenes N] [--seed S] [--no-resume] [--allow-mixed] [--json-progress]
python -m syngt export DATASET [--formats coco coco_bb yolo_obb ...] [--min-visib 0.3] [--min-px 50] [--max-trunc 0.5]
                               [--obb-source visible_mask|amodal_mask|projected_3d_box]
python -m syngt preview cfg.yaml --scene 3 --out preview_dir
python -m syngt presets
```

`tests/test_consistency.py` checks renderer/GT consistency for every placement mode, a distorted lens, a fixed
2-camera rig and independent cameras. It projects CAD vertices with the GT pose and intrinsics into the
rendered amodal masks and checks depth against the poses.

## Limits and notes

* Rasterised PBR (GGX), not path tracing: no refraction, inter-reflections or caustics. Specular metal gets a
  hemisphere environment term. Expect to close the gap with sensor/lighting randomisation and real backdrops.
* Depth is rendered *aligned to the colour camera*, as with `rs.align` or Orbbec D2C. Stereo shadows are
  approximated by edge bands.
* Kalibr `equidistant` (fisheye) models are imported without distortion. OpenCV rational/tangential models
  up to k6 are fully supported.
* IGES is not supported (export STEP/STL/OBJ/GLB instead). Mesh processing results (stable poses, hulls) are
  cached in `%TEMP%\syngt_cache` (*Tools → Clear mesh cache*).
* Scale jitter ≠ 1 changes the object, so the GT pose then refers to the scaled model (`scale` is stored).
