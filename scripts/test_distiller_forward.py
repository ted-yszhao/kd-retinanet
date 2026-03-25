import torch
from torch.utils.data import DataLoader

from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.models.retinanet_builder import build_teacher, build_student
from src.models.distiller import RetinaNetDistiller
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

images, targets = next(iter(loader))
images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

teacher = build_teacher().to(device)
student = build_student().to(device)
kd_loss = MultiScaleSSIMKD(keys=("2",), weights=(1.0,)).to(device)

model = RetinaNetDistiller(
    teacher=teacher,
    student=student,
    kd_loss=kd_loss,
    kd_weight=0.5,
).to(device)

model.train()
losses = model(images, targets)

for k, v in losses.items():
    print(k, float(v.detach()))