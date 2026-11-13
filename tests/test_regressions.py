"""One test per bug that was found and fixed (physics, assets, viewpoints, camera import).

Run:  python -m pytest tests/test_regressions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from syngt.assets import load_asset, model_extent, symmetry_info, unit_scale  # noqa: E402
from syngt.cameras import _cam, camera_from_preset, load_calibration, save_calibration_json  # noqa: E402
from syngt.config import IntRange, ObjectModel, ProjectConfig, R  # noqa: E402
from syngt.geometry import transform_points  # noqa: E402
from syngt.scene import SceneSampler  # noqa: E402
from syngt.viewpoints import sample_scene_views  # noqa: E402

MODELS = ROOT / "examples" / "models"


def cfg_with(*names, count=(2, 2)) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.objects.distractors.enabled = False
    for i, n in enumerate(names, 1):
        cfg.objects.models.append(ObjectModel(path=str(MODELS / f"{n}.stl"), name=n, class_id=i, unit="mm",
                                              count=IntRange(min=count[0], max=count[1])))
    return cfg


def lowest_z(st):
    return [transform_points(a["T_world"], a["asset"].vertices * a["scale"])[:, 2].min() for a in st.annotated]


def cam(name="a"):
    return camera_from_preset("Intel RealSense D435 / D435i", "color", 640, 480, name=name)


# ------------------------------------------------------------------ physics / placement
def test_parts_rest_on_the_table_not_above_it():
    """A contact margin made every part hover 1 mm above the table (a third of a LEGO tile's thickness)."""
    cfg = cfg_with("brick_50x10x5", count=(4, 4))
    cfg.placement.cluster_radius = R(0.08, 0.08)
    z = lowest_z(SceneSampler(cfg).sample_scene(0))
    assert z and max(abs(v) for v in z) < 2e-4, z


def test_part_with_far_away_cad_origin_is_placed_like_any_other(tmp_path):
    """Bounding radius / out-of-area test were measured from the CAD origin: off-origin parts were thrown out."""
    m = trimesh.creation.box(extents=(0.03, 0.02, 0.01))
    m.apply_translation((0.15, 0.0, 0.08))
    f = tmp_path / "offset_part.stl"
    m.export(f)
    for mode in ("physics_drop", "free_floating"):
        cfg = ProjectConfig()
        cfg.objects.distractors.enabled = False
        cfg.objects.models.append(ObjectModel(path=str(f), name="p", unit="m", count=IntRange(min=6, max=6)))
        cfg.placement.mode = mode
        s = SceneSampler(cfg)
        placed = [s.sample_scene(i).info["n_placed"] for i in range(3)]
        assert min(placed) >= 5, (mode, placed)


def test_floating_parts_never_cut_through_the_table():
    cfg = cfg_with("gear", count=(4, 4))
    cfg.placement.mode = "free_floating"
    s = SceneSampler(cfg)
    assert min(z for i in range(8) for z in lowest_z(s.sample_scene(i))) > 0


def test_total_mode_with_zero_weight_does_not_crash():
    cfg = cfg_with("gear", "bracket", "flange")
    cfg.objects.models[2].weight = 0.0
    cfg.objects.count_mode, cfg.objects.min_classes_per_scene = "total", 3
    st = SceneSampler(cfg).sample_scene(0)
    assert st.info["n_placed"] > 0 and all(a["model_name"] != "flange" for a in st.annotated)


def test_zero_counts_give_background_only_scenes():
    cfg = cfg_with("gear", count=(0, 0))
    assert SceneSampler(cfg).sample_scene(0).info["n_placed"] == 0


# ------------------------------------------------------------------ assets
def test_stl_is_smooth_shaded(tmp_path):
    """STL is a triangle soup: without welding the vertices 'shading crease angle' did nothing."""
    f = tmp_path / "sphere.stl"
    trimesh.creation.icosphere(subdivisions=3).export(f)
    part = load_asset(str(f), unit="m").parts[0]
    radial = part.positions / np.linalg.norm(part.positions, axis=1, keepdims=True)
    err = np.degrees(np.arccos(np.clip((radial * part.normals).sum(1), -1, 1))).mean()
    assert len(part.positions) < 1000 and err < 1.0


def test_two_models_of_one_file_keep_their_names():
    a = load_asset(str(MODELS / "gear.stl"), name="gear_red")
    b = load_asset(str(MODELS / "gear.stl"), name="gear_blue")
    assert (a.name, b.name) == ("gear_red", "gear_blue") and a.key == b.key


def test_step_files_are_not_scaled_twice():
    assert unit_scale("part.STEP", "mm", 0.05) == ("m", 1.0)       # the loader already returns metres
    assert unit_scale("part.stl", "mm", 50.0) == ("mm", 0.001)
    assert unit_scale("part.stl", "auto", 50.0)[0] == "mm" and unit_scale("part.stl", "auto", 0.05)[0] == "m"
    assert model_extent(str(MODELS / "brick_50x10x5.stl"), "mm") == pytest.approx(0.05, abs=1e-4)


def test_symmetry_order_zero_gives_finite_matrices():
    a = load_asset(str(MODELS / "gear.stl"))
    assert np.isfinite(np.array(symmetry_info("discrete_z", 0, a)["symmetries_discrete"])).all()


# ------------------------------------------------------------------ viewpoints
def scene_points():
    return np.array([0.01, 0.02, 0.01]), np.random.default_rng(0).normal(0, 0.03, (5, 3))


def test_orbit_is_a_circle():
    v = ProjectConfig().viewpoints
    v.mode, v.views_per_scene = "orbit", 6
    c, oc = scene_points()
    eyes = np.array([T[0][:3, 3] for T in sample_scene_views(v, [cam()], c, 0.05, oc, np.random.default_rng(1))])
    centre = eyes.mean(0)
    assert np.ptp(np.linalg.norm(eyes[:, :2] - centre[:2], axis=1)) < 1e-6 and np.ptp(eyes[:, 2]) < 1e-6


def test_fixed_rig_keeps_its_baseline_under_jitter():
    v = ProjectConfig().viewpoints
    v.mode, v.fixed_jitter_translation, v.fixed_jitter_rotation_deg = "fixed", 0.01, 2.0
    cams = [cam("a"), cam("b")]
    cams[1].rig_translation = [0.06, 0.0, 0.0]
    c, oc = scene_points()
    views = sample_scene_views(v, cams, c, 0.05, oc, np.random.default_rng(2))
    base = [np.linalg.norm(T[0][:3, 3] - T[1][:3, 3]) for T in views]
    assert np.allclose(base, 0.06, atol=1e-9)
    assert np.ptp([T[0][:3, 3] for T in views], axis=0).max() > 1e-4       # ... while the rig itself is jittered


def test_swapped_elevation_range_does_not_crash():
    cfg = cfg_with("gear")
    cfg.viewpoints.mode, cfg.viewpoints.elevation_deg = "sphere_random", R(90, 25)
    assert len(SceneSampler(cfg).sample_scene(0).views) == cfg.viewpoints.views_per_scene


def test_no_camera_is_a_clear_error():
    cfg = cfg_with("gear")
    cfg.cameras = []
    with pytest.raises(ValueError, match="camera"):
        SceneSampler(cfg)


# ------------------------------------------------------------------ camera calibration files
def test_calibration_json_round_trip_keeps_the_rig(tmp_path):
    cams = [cam("a"), cam("b")]
    cams[1].rig_translation, cams[1].rig_rotation_deg = [0.05, 0.0, 0.0], [0.0, 10.0, 0.0]
    f = tmp_path / "cams.json"
    save_calibration_json(cams, str(f))
    back = load_calibration(str(f))
    assert back[1].rig_translation == [0.05, 0.0, 0.0] and back[1].rig_rotation_deg == [0.0, 10.0, 0.0]


def test_six_distortion_coefficients_are_padded():
    assert len(_cam("x", np.eye(3), [0.1] * 6, 10, 10).dist) == 8


def test_windows_style_relative_paths_load_on_linux(tmp_path):
    if os.name == "nt":
        pytest.skip("only relevant off Windows")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "p.stl").write_text("x")
    (tmp_path / "cfg.yaml").write_text("objects:\n  models:\n  - path: models\\p.stl\n")
    assert ProjectConfig.load(tmp_path / "cfg.yaml").objects.models[0].path == str(tmp_path / "models" / "p.stl")


# ------------------------------------------------------------------ annotations / export
def test_truncation_counts_what_lies_beyond_the_render_canvas():
    """Truncation was measured only inside the padded canvas: close-ups were kept although mostly outside."""
    def run(padding):
        cfg = cfg_with("gear", "flange", count=(3, 3))
        cfg.cameras = [cam()]
        cfg.viewpoints.radius, cfg.viewpoints.views_per_scene = R(0.07, 0.10), 6     # close-ups
        cfg.viewpoints.min_visible_objects = 0
        cfg.sensor.enabled = False
        cfg.render.canvas_padding = padding
        s = SceneSampler(cfg)
        out = {}
        for i in range(3):
            st = s.sample_scene(i)
            for v in range(len(st.views)):
                for a in s.render_view(st, v, np.random.default_rng([i, v]))[0].annotations:
                    out[(i, v, a["inst_id"])] = a["truncation"]
        return out
    truth, small, none = run(4.0), run(0.15), run(0.0)      # 4.0: canvas 9x the image = ground truth
    for est in (small, none):
        keys = [k for k in truth if k in est and truth[k] > 0.3]
        assert len(keys) > 10
        err = np.array([abs(est[k] - truth[k]) for k in keys])
        assert np.median(err) < 0.08 and err.max() < 0.3, (np.median(err), err.max())
    # the filter must work without any padding too (it used to see truncation 0 everywhere)
    assert sum(none[k] > 0.6 for k in none) >= 0.7 * sum(truth[k] > 0.6 for k in none if k in truth)


def test_single_polygon_keeps_holes_and_pieces():
    import cv2
    from syngt.annotations import mask_has_holes, mask_polygons, single_polygon
    m = np.zeros((120, 160), np.uint8)
    cv2.circle(m, (50, 60), 40, 1, -1)
    cv2.circle(m, (50, 60), 18, 0, -1)          # a ring (hex nut, flange ...)
    cv2.rectangle(m, (110, 20), (150, 50), 1, -1)   # a second piece (object split by an occluder)
    assert mask_has_holes(m)
    poly = np.array(single_polygon(m, tol=0.5)).reshape(-1, 2)
    r = np.zeros_like(m)
    cv2.fillPoly(r, [np.round(poly).astype(np.int32)], 1)
    iou = (r & m).sum() / (r | m).sum()
    assert iou > 0.95, iou
    # the old export took only the largest outer outline: hole filled, second piece lost
    old = np.zeros_like(m)
    cv2.fillPoly(old, [np.array(max(mask_polygons(m, 0.5), key=len), np.float32).reshape(-1, 2).astype(np.int32)], 1)
    assert (old & m).sum() / (old | m).sum() < 0.8


def test_obb_at_the_image_border_stays_a_rectangle_inside_the_image():
    from syngt.writers import _obb_in_image
    c = np.array([[-21.4, 303.2], [40.0, 250.0], [80.0, 296.0], [18.6, 349.2]])   # rotated box sticking out left
    b = _obb_in_image(c, 640, 480)
    assert b.min() >= 0 and b[:, 0].max() <= 640 and b[:, 1].max() <= 480
    d = [np.linalg.norm(b[i] - b[(i + 1) % 4]) for i in range(4)]
    assert abs(d[0] - d[2]) < 1e-3 and abs(d[1] - d[3]) < 1e-3                    # still a rectangle
    assert abs(np.dot(b[1] - b[0], b[2] - b[1])) < 1e-2 * d[0] * d[1] + 1e-6
    inside = np.array([[10.0, 10.0], [50, 10], [50, 30], [10, 30]])
    assert np.allclose(_obb_in_image(inside, 640, 480), inside)                   # untouched when inside


def test_transparent_output_has_straight_colour_at_the_edges():
    """Edge pixels of RGBA images were stored premultiplied (dark halos in any normal compositor)."""
    for gpu in (True, False):
        cfg = cfg_with("gear", count=(2, 2))
        cfg.cameras = [cam()]
        cfg.scene.background.mode, cfg.scene.ground.enabled = "transparent", False
        cfg.placement.mode = "free_floating"
        cfg.sensor.enabled, cfg.render.gpu_postprocess = False, gpu
        cfg.scene.lighting.ambient, cfg.scene.lighting.num_lights = R(3, 3), IntRange(min=0, max=0)   # flat light
        s = SceneSampler(cfg)
        img = s.render_view(s.sample_scene(0), 0, np.random.default_rng(0))[0].rgb
        assert img.shape[2] == 4
        a = img[..., 3].astype(float) / 255
        lum = img[..., :3].astype(float).mean(2)
        inner, edge = lum[a > 0.99], lum[(a > 0.3) & (a < 0.7)]
        assert len(edge) > 20 and edge.mean() > 0.85 * inner.mean(), (gpu, edge.mean(), inner.mean())


def test_depth_beyond_16_bit_is_invalid_not_wrong(tmp_path):
    from syngt.pipeline import generate
    cfg = cfg_with("gear", count=(1, 1))
    cfg.cameras = [cam()]
    cfg.viewpoints.views_per_scene, cfg.generation.num_scenes = 1, 1
    cfg.viewpoints.elevation_deg = R(10, 12)                    # low view: the table runs to the horizon
    cfg.output.save_depth, cfg.output.depth_scale_mm, cfg.output.formats = True, 0.01, ["native"]   # range 0.655 m
    cfg.cameras[0].depth.model, cfg.cameras[0].depth.max_z = "ideal", 10.0
    generate(cfg, str(tmp_path / "d"), lambda e: None, workers=1)
    import cv2
    d = cv2.imread(str(next((tmp_path / "d" / "depth").rglob("*.png"))), cv2.IMREAD_UNCHANGED)
    # saturation used to pile hundreds of far pixels onto 65535 (a wrong distance); now they are 0 = invalid
    assert d.dtype == np.uint16 and (d == 65535).sum() < 5 and (d == 0).sum() > 100


def test_checks_warn_about_what_bop_cannot_express():
    from syngt.validation import check_config
    cfg = cfg_with("gear")
    cfg.output.formats = ["native", "bop"]
    assert not any("BOP" in i.message for i in check_config(cfg))
    cfg.objects.models[0].scale_jitter = R(0.8, 1.2)
    cfg.cameras[0].dist = [-0.2, 0.05, 0, 0, 0]
    msgs = [i.message for i in check_config(cfg)]
    assert any("scale jitter" in m for m in msgs) and any("lens distortion" in m for m in msgs)
    cfg.cameras.append(cfg.cameras[0].model_copy(update={"name": cfg.cameras[0].name.replace("_", " ") + "/"}))
    cfg.cameras[0].name = cfg.cameras[1].name.replace("/", "_")
    assert any("same file name" in i.message for i in check_config(cfg))


# ------------------------------------------------------------------ look profiles / real-photo tools
def test_look_profile_auto_exposure_reproduces_the_tuned_exposure():
    """The profile's target meant 'frame brightness' but was applied to a grey card: wrong brightness."""
    from syngt.imaging import display_brightness
    from syngt.lookprofile import apply_profile, light_metering_target
    from syngt.scene import GREY_CARD, sample_lights, white_radiance
    cfg = ProjectConfig()
    prof = {"name": "old", "sensor": {"exposure": {"min": 0.6, "max": 0.9, "dist": "loguniform"},
                                      "auto_exposure": True, "ae_target": {"min": 0.45, "max": 0.52}},
            "lighting": cfg.scene.lighting.model_dump(mode="json")}
    cfg.sensor.enabled = False
    cfg.scene.lighting.shadows, cfg.scene.lighting.shadow_map_size = False, 4096
    apply_profile(cfg, prof)
    assert cfg.sensor.enabled and cfg.sensor.ae_metering == "light"
    assert (cfg.scene.lighting.shadows, cfg.scene.lighting.shadow_map_size) == (False, 4096)   # not part of a look
    t = light_metering_target(cfg.sensor.exposure, cfg.scene.lighting)
    assert (cfg.sensor.ae_target.min, cfg.sensor.ae_target.max) == (t.min, t.max)
    # the grey card under typical lighting, exposed with the tuned exposure, lands inside the target range
    rng = np.random.default_rng(1)
    g = np.exp(np.mean([np.log(white_radiance(*sample_lights(cfg.scene.lighting, rng, np.zeros(3))[:2], np.zeros(3))
                               * GREY_CARD) for _ in range(300)]))
    mid = display_brightness(np.array([[[np.sqrt(0.6 * 0.9) * g] * 3]], np.float32), "filmic")
    assert t.min - 0.03 <= mid <= t.max + 0.03
    # a profile that already stores its metering mode is taken as it is
    prof["sensor"]["ae_metering"] = "frame"
    apply_profile(cfg, prof)
    assert cfg.sensor.ae_metering == "frame" and cfg.sensor.ae_target.min == 0.45


def test_broken_look_profile_files_are_rejected_cleanly(tmp_path):
    from syngt.lookprofile import load_profile
    for text in ("[1, 2, 3]", '{"name": "x"}'):
        f = tmp_path / "bad.json"
        f.write_text(text)
        with pytest.raises(ValueError):
            load_profile(str(f))


def test_match_look_adjusts_what_auto_exposure_actually_uses():
    from syngt.appearance import apply_to_config
    stats = lambda b: {k: {"mean": v, "p10": v, "p90": v} for k, v in  # noqa: E731
                       dict(brightness=b, contrast=0.2, saturation=0.2, cast_r=1.0, cast_b=1.0, noise=0.004,
                            sharpness=1.0).items()}
    cfg = ProjectConfig()
    t0, e0 = cfg.sensor.ae_target.max, cfg.sensor.exposure.max
    apply_to_config(cfg, stats(0.6), stats(0.3))
    assert cfg.sensor.ae_target.max > t0 and cfg.sensor.exposure.max == e0      # auto exposure on: target moves
    cfg = ProjectConfig()
    cfg.sensor.auto_exposure = False
    apply_to_config(cfg, stats(0.6), stats(0.3))
    assert cfg.sensor.exposure.max > e0 and cfg.sensor.exposure.max <= 100


def test_image_folders_with_brackets_are_found(tmp_path):
    import cv2
    from syngt.appearance import list_images
    d = tmp_path / "real photos [cam1]"
    d.mkdir()
    cv2.imwrite(str(d / "a.png"), np.zeros((8, 8, 3), np.uint8))
    assert len(list_images(str(d))) == 1


def test_overwritten_texture_file_is_reloaded(tmp_path):
    import cv2
    import time
    from syngt.textures import load_image_rgb
    f = tmp_path / "t.png"
    cv2.imwrite(str(f), np.full((8, 8, 3), (0, 0, 255), np.uint8))      # red (BGR)
    assert load_image_rgb(str(f))[0, 0].tolist() == [255, 0, 0]
    time.sleep(0.02)
    cv2.imwrite(str(f), np.full((9, 9, 3), (255, 0, 0), np.uint8))      # blue
    assert load_image_rgb(str(f))[0, 0].tolist() == [0, 0, 255]


def test_white_parts_are_not_blown_out_and_do_not_depend_on_the_table():
    """Light-metered auto exposure: no clipped parts, and part brightness is the same on a black and a white table."""
    def run(table_rgb):
        cfg = cfg_with("brick_50x10x5", count=(4, 4))
        cfg.cameras = [cam()]
        cfg.objects.models[0].material.color = [0.96, 0.96, 0.96]          # white plastic
        cfg.scene.ground.texture_mode, cfg.scene.ground.palette = "palette", [table_rgb]
        cfg.sensor.ae_target = R(0.65, 0.65)                                 # a bright target that would clip white
        for r in (cfg.sensor.vignetting, cfg.sensor.gaussian_noise, cfg.sensor.shot_noise):
            r.min = r.max = 0.0
        cfg.sensor.defocus_prob = cfg.sensor.motion_blur_prob = 0.0
        s = SceneSampler(cfg)
        vals, clipped, limited = [], [], []
        for i in range(4):
            fr = s.render_view(s.sample_scene(i), 0, np.random.default_rng(i))[0]
            px = fr.rgb[..., :3][fr.inst > 0].astype(float)
            vals.append(px.mean())
            clipped.append((px.min(1) >= 252).mean())
            limited.append(fr.sensor_params["ae_highlight_limited"])
        assert any(limited)             # the bright target needed the highlight limit at least once
        return np.mean(vals), np.mean(clipped)
    dark, light = run([0.05, 0.05, 0.05]), run([0.9, 0.9, 0.9])
    assert dark[1] < 0.01 and light[1] < 0.01                               # no pure-white part pixels
    assert abs(dark[0] - light[0]) < 0.12 * dark[0], (dark, light)          # table colour does not move the parts
