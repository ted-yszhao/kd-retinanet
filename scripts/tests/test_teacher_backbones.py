from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.retinanet_builder import build_teacher


def assert_teacher_builds(backbone_name: str) -> None:
    model = build_teacher(
        backbone_name=backbone_name,
        backbone_weights=None,
    )
    assert model.backbone.out_channels == 256


assert_teacher_builds("resnet50")
assert_teacher_builds("resnet101")

print("teacher backbone builder smoke test passed")
