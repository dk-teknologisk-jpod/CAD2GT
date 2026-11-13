"""Write examples/example_config.yaml: 6D pose of small bricks (50 x 10 x 5 mm), RealSense D435.

Everything not set here uses the library defaults (syngt/config.py), which are tuned for
small parts on a table seen from 0.3-0.6 m, with native + BOP output.
"""
from pathlib import Path

from syngt.config import MaterialConfig, ObjectModel, ProjectConfig, R

here = Path(__file__).parent
cfg = ProjectConfig(name="small_bricks_pose")
cfg.objects.models.append(ObjectModel(
    path="models/brick_50x10x5.stl", name="brick", class_id=1, unit="mm",
    material=MaterialConfig(mode="fixed_color", color=[0.80, 0.12, 0.10], hue_jitter=0.02,
                            saturation_jitter=0.1, value_jitter=0.2, roughness=R(0.25, 0.6), metallic=R(0.0, 0.05)),
    symmetry="box_xyz"))
cfg.output.output_dir = "../output/bricks_pose"
from syngt.lookprofile import apply_profile, default_profile, load_profile  # noqa: E402
if default_profile():  # camera look measured from real Orbbec Gemini 335 images
    apply_profile(cfg, load_profile(default_profile()), default_profile())
cfg.save(here / "example_config.yaml")
print("written", here / "example_config.yaml")
