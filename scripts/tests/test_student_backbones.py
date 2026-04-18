from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.retinanet_builder import build_student


def assert_student_builds(backbone_name: str) -> None:
    model = build_student(
        backbone_name=backbone_name,
        backbone_weights=None,
        detector_weights=None,
    )
    assert model.backbone.out_channels == 256


assert_student_builds("resnet18")
assert_student_builds("resnet50")

print("student backbone builder smoke test passed")
