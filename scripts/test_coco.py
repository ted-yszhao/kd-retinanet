from torchvision.datasets import CocoDetection

root = "datasets/coco/val2017"
ann = "datasets/coco/annotations/instances_val2017.json"

dataset = CocoDetection(root=root, annFile=ann)

img, target = dataset[0]

print(type(img))
print(type(target))
print("num anns:", len(target))
print("first ann keys:", target[0].keys() if len(target) > 0 else "no ann")