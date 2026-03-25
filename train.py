import torch
from torch.utils.data import DataLoader
from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.models.retinanet_builder import build_teacher, build_student
from src.models.distiller import RetinaNetDistiller
from src.kd.ssim_kd import MultiScaleSSIMKD
from src.engine.train_one_epoch import train_one_epoch
from utils.logger import TrainLogger

device = "cuda" if torch.cuda.is_available() else "cpu"


run_name = "retinanet_coco_run1"
logger = TrainLogger(run_dir=f"runs/{run_name}")

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

teacher = build_teacher().to(device)
student = build_student().to(device)
kd_loss = MultiScaleSSIMKD(keys=("2",), weights=(1.0,)).to(device)

model = RetinaNetDistiller(
    teacher=teacher,
    student=student,
    kd_loss=kd_loss,
    kd_weight=0.5,
).to(device)

optimizer = torch.optim.SGD(
    [p for p in model.student.parameters() if p.requires_grad],
    lr=0.001,
    momentum=0.9,
    weight_decay=1e-4,
)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    stats = train_one_epoch(model, loader, optimizer, device)
    print(f"Avg Losses: {stats}")