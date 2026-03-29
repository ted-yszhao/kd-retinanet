import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn

from src.data.coco import CocoDetectionKD, detection_collate_fn

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

model = retinanet_resnet50_fpn(weights=None, num_classes=91).to(device)
model.train()

loss_dict = model(images, targets)

print("loss keys:", loss_dict.keys())
for k, v in loss_dict.items():
    print(k, float(v.detach()))