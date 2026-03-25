from src.data.coco import CocoDetectionKD

dataset = CocoDetectionKD(
    root="datasets/coco/val2017",
    annFile="datasets/coco/annotations/instances_val2017.json",
)

img, target = dataset[0]

print("image shape:", img.shape)
print("boxes shape:", target["boxes"].shape)
print("labels shape:", target["labels"].shape)
print("target keys:", target.keys())