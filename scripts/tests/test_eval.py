import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.detection import evaluate_detection_model
from src.models.distiller import RetinaNetDistiller


class ToyDetectionDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        image = torch.zeros((3, 32, 32), dtype=torch.float32)
        target = {
            "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([256.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class PerfectDetector(nn.Module):
    def forward(self, images, targets=None):
        if targets is not None:
            return {"loss_classifier": torch.tensor(0.0), "loss_box_reg": torch.tensor(0.0)}

        outputs = []
        for _ in images:
            outputs.append({
                "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.99], dtype=torch.float32),
            })
        return outputs


class ZeroKDLoss(nn.Module):
    def forward(self, student_feats, teacher_feats):
        return torch.tensor(0.0)


def main():
    dataset = ToyDetectionDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    teacher = PerfectDetector()
    student = PerfectDetector()
    kd_model = RetinaNetDistiller(
        teacher=teacher,
        student=student,
        kd_loss=ZeroKDLoss(),
        kd_weight=0.5,
    )

    teacher_metrics = evaluate_detection_model(teacher, loader, device="cpu")
    student_metrics = evaluate_detection_model(student, loader, device="cpu")
    kd_metrics = evaluate_detection_model(kd_model, loader, device="cpu", branch="full")

    print("teacher:", teacher_metrics)
    print("student:", student_metrics)
    print("full kd:", kd_metrics)

    assert teacher_metrics["map_50"] > 0.99
    assert student_metrics["map_50"] > 0.99
    assert kd_metrics["map_50"] > 0.99


if __name__ == "__main__":
    main()
