from __future__ import annotations

import torch


def _unwrap_dataset(dataset):
    current = dataset
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def _get_coco_api_from_loader(loader):
    dataset = _unwrap_dataset(loader.dataset)
    coco = getattr(dataset, "coco", None)
    if coco is None:
        raise ValueError("The evaluation loader dataset must expose a pycocotools COCO API via dataset.coco.")
    return coco


def _prepare_output_dict(output: dict, image_id: int) -> dict:
    return {
        "image_id": image_id,
        "boxes": output.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().to(torch.float32),
        "scores": output.get("scores", torch.zeros((0,), dtype=torch.float32)).detach().cpu().to(torch.float32),
        "labels": output.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().to(torch.int64),
    }


def _collect_predictions(model, loader, device, branch: str = "student") -> list[dict]:
    model_device = torch.device(device)
    outputs = []

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for images, batch_targets in loader:
            images = [image.to(model_device) for image in images]

            if hasattr(model, "teacher") and hasattr(model, "student"):
                batch_outputs = model(images, targets=None, branch=branch)
            else:
                batch_outputs = model(images)

            outputs.extend(
                _prepare_output_dict(output, int(target["image_id"].flatten()[0].item()))
                for output, target in zip(batch_outputs, batch_targets)
            )

    if was_training:
        model.train()

    return outputs


def _prediction_to_coco_results(predictions: list[dict]) -> list[dict]:
    results = []
    for prediction in predictions:
        image_id = prediction["image_id"]
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        labels = prediction["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score.item()),
                }
            )
    return results


def evaluate_detection_model(model, loader, device="cpu", branch: str = "student") -> dict[str, float]:
    """
    Evaluate a detector with pycocotools COCOeval.

    For RetinaNetDistiller:
    - branch="student" or "full" evaluates the student detector
    - branch="teacher" evaluates the frozen teacher detector
    """
    from pycocotools.cocoeval import COCOeval

    coco_gt = _get_coco_api_from_loader(loader)
    predictions = _collect_predictions(model=model, loader=loader, device=device, branch=branch)
    coco_results = _prediction_to_coco_results(predictions)

    if not coco_results:
        return {
            "bbox_mAP": 0.0,
            "bbox_mAP_50": 0.0,
            "bbox_mAP_75": 0.0,
            "bbox_mAP_small": 0.0,
            "bbox_mAP_medium": 0.0,
            "bbox_mAP_large": 0.0,
            "num_images": float(len(predictions)),
        }

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted({prediction["image_id"] for prediction in predictions})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "bbox_mAP": float(coco_eval.stats[0]),
        "bbox_mAP_50": float(coco_eval.stats[1]),
        "bbox_mAP_75": float(coco_eval.stats[2]),
        "bbox_mAP_small": float(coco_eval.stats[3]),
        "bbox_mAP_medium": float(coco_eval.stats[4]),
        "bbox_mAP_large": float(coco_eval.stats[5]),
        "num_images": float(len(predictions)),
    }
