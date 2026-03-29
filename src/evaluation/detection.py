from __future__ import annotations

from collections import defaultdict

import torch


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - intersection

    return intersection / union.clamp(min=1e-6)


def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    for index in range(precisions.numel() - 1, 0, -1):
        precisions[index - 1] = torch.maximum(precisions[index - 1], precisions[index])

    recall_levels = torch.linspace(0.0, 1.0, steps=101)
    interpolated = []
    for level in recall_levels:
        valid = precisions[recalls >= level]
        interpolated.append(valid.max() if valid.numel() else torch.tensor(0.0))

    return float(torch.stack(interpolated).mean())


def _prepare_output_dict(output: dict) -> dict:
    return {
        "boxes": output.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().to(torch.float32),
        "scores": output.get("scores", torch.zeros((0,), dtype=torch.float32)).detach().cpu().to(torch.float32),
        "labels": output.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().to(torch.int64),
    }


def _prepare_target_dict(target: dict) -> dict:
    return {
        "boxes": target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().to(torch.float32),
        "labels": target.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().to(torch.int64),
        "image_id": int(target["image_id"].flatten()[0].item()),
    }


def _collect_predictions(model, loader, device, branch: str = "student") -> tuple[list[dict], list[dict]]:
    model_device = torch.device(device)
    outputs = []
    targets = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for images, batch_targets in loader:
            images = [image.to(model_device) for image in images]

            if hasattr(model, "teacher") and hasattr(model, "student"):
                batch_outputs = model(images, targets=None, branch=branch)
            else:
                batch_outputs = model(images)

            outputs.extend(_prepare_output_dict(output) for output in batch_outputs)
            targets.extend(_prepare_target_dict(target) for target in batch_targets)

    if was_training:
        model.train()

    return outputs, targets


def _evaluate_at_iou(predictions: list[dict], targets: list[dict], iou_threshold: float) -> float:
    classes = sorted({
        int(label.item())
        for target in targets
        for label in target["labels"]
    } | {
        int(label.item())
        for prediction in predictions
        for label in prediction["labels"]
    })

    if not classes:
        return 0.0

    target_lookup = {}
    gt_count_by_class = defaultdict(int)
    for target in targets:
        image_id = target["image_id"]
        target_lookup[image_id] = target
        for class_id in target["labels"].tolist():
            gt_count_by_class[int(class_id)] += 1

    ap_values = []
    for class_id in classes:
        gt_matches = {}
        detections = []

        for target in targets:
            image_id = target["image_id"]
            mask = target["labels"] == class_id
            boxes = target["boxes"][mask]
            gt_matches[image_id] = torch.zeros((boxes.shape[0],), dtype=torch.bool)

        for prediction, target in zip(predictions, targets):
            image_id = target["image_id"]
            mask = prediction["labels"] == class_id
            for box, score in zip(prediction["boxes"][mask], prediction["scores"][mask]):
                detections.append({
                    "image_id": image_id,
                    "box": box,
                    "score": float(score.item()),
                })

        num_gt = gt_count_by_class.get(class_id, 0)
        if num_gt == 0:
            continue

        detections.sort(key=lambda detection: detection["score"], reverse=True)

        if not detections:
            ap_values.append(0.0)
            continue

        tps = torch.zeros((len(detections),), dtype=torch.float32)
        fps = torch.zeros((len(detections),), dtype=torch.float32)

        for index, detection in enumerate(detections):
            image_id = detection["image_id"]
            target = target_lookup[image_id]
            mask = target["labels"] == class_id
            gt_boxes = target["boxes"][mask]

            if gt_boxes.numel() == 0:
                fps[index] = 1.0
                continue

            ious = _box_iou(detection["box"].unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_index = ious.max(dim=0)

            if best_iou >= iou_threshold and not gt_matches[image_id][best_index]:
                tps[index] = 1.0
                gt_matches[image_id][best_index] = True
            else:
                fps[index] = 1.0

        tp_cum = torch.cumsum(tps, dim=0)
        fp_cum = torch.cumsum(fps, dim=0)
        recalls = tp_cum / max(num_gt, 1)
        precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)
        ap_values.append(_compute_ap(recalls, precisions))

    return float(sum(ap_values) / len(ap_values)) if ap_values else 0.0


def evaluate_detection_model(model, loader, device="cpu", branch: str = "student") -> dict[str, float]:
    """
    Evaluate a detector on a detection dataloader and return common AP metrics.

    For RetinaNetDistiller:
    - branch="student" or "full" evaluates the student detector
    - branch="teacher" evaluates the frozen teacher detector
    """
    predictions, targets = _collect_predictions(model=model, loader=loader, device=device, branch=branch)

    thresholds = [round(value, 2) for value in torch.arange(0.5, 1.0, 0.05).tolist()]
    ap_by_threshold = {
        threshold: _evaluate_at_iou(predictions=predictions, targets=targets, iou_threshold=threshold)
        for threshold in thresholds
    }

    return {
        "map": float(sum(ap_by_threshold.values()) / len(ap_by_threshold)) if ap_by_threshold else 0.0,
        "map_50": ap_by_threshold.get(0.5, 0.0),
        "map_75": ap_by_threshold.get(0.75, 0.0),
        "num_images": float(len(targets)),
    }
