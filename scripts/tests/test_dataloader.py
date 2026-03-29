from torch.utils.data import DataLoader
from src.data.coco import CocoDetectionKD, detection_collate_fn

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

print("batch size:", len(images))
print("image 0 shape:", images[0].shape)
print("target 0 keys:", targets[0].keys())
print("target 0 boxes:", targets[0]["boxes"].shape)
