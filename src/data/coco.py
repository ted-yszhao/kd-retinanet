import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor


class CocoDetectionKD(CocoDetection):
    """
    A wrapper around torchvision's CocoDetection to convert annotations into the format expected by RetinaNet.
    It converts bounding boxes from [x, y, w, h] to [x_min, y_min, x_max, y_max] and creates tensors for boxes, labels, areas, and iscrowd flags.
    """
    def __init__(self, root, annFile):
        super().__init__(root=root, annFile=annFile)

    def __getitem__(self, idx):
        img, ann = super().__getitem__(idx)
        image_id = self.ids[idx]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in ann:
            x, y, w, h = obj["bbox"]
            if w <= 1 or h <= 1:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])
            areas.append(obj.get("area", w * h))
            iscrowd.append(obj.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        img = pil_to_tensor(img).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return img, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
