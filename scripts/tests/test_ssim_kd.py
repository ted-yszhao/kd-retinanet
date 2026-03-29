import torch
from torch.utils.data import DataLoader

from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.models.retinanet_builder import build_teacher, build_student
from src.models.feature_extractor import extract_fpn_feats
from src.kd.ssim_kd import MultiScaleSSIMKD

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

images, _ = next(iter(loader))
images = [img.to(device) for img in images]

teacher = build_teacher().to(device).eval()
student = build_student().to(device).eval()

# 
kd_loss = MultiScaleSSIMKD(keys=("2",), weights=(1.0,)).to(device)

with torch.no_grad():
    t_feats = extract_fpn_feats(teacher, images)
s_feats = extract_fpn_feats(student, images)

loss = kd_loss(s_feats, t_feats)
print("kd loss:", float(loss.detach()))