import torch
from torch.utils.data import DataLoader

from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.models.retinanet_builder import build_teacher, build_student
from src.models.feature_extractor import extract_fpn_feats

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CocoDetectionKD(
    root="datasets/coco/val2017",
    annFile="datasets/coco/annotations/instances_val2017.json",
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn=detection_collate_fn,
)

images, targets = next(iter(loader))
images = [img.to(device) for img in images]

teacher = build_teacher().to(device).eval()
student = build_student().to(device).eval()

with torch.no_grad():
    t_feats = extract_fpn_feats(teacher, images)
    s_feats = extract_fpn_feats(student, images)

print("Teacher FPN shapes:")
for k, v in t_feats.items():
    print(k, tuple(v.shape))

print("Student FPN shapes:")
for k, v in s_feats.items():
    print(k, tuple(v.shape))