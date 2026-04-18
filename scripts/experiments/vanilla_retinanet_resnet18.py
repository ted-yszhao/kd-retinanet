import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.coco import CocoDetectionKD
from src.evaluation.detection import evaluate_detection_model
from src.models.retinanet_builder import build_student
from train import build_detection_loader, build_epoch_loader, build_scheduler, resolve_device


def build_parser():
    parser = argparse.ArgumentParser(description="Train a vanilla RetinaNet detector from config.")
    parser.add_argument(
        "--config",
        default="configs/vanilla_retinanet_resnet18.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, for example --set optimizer.lr=0.001",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    return parser


def train_vanilla_from_config(config):
    device = resolve_device(config.training.device)

    train_dataset = CocoDetectionKD(
        root=config.data.root,
        annFile=config.data.ann_file,
    )

    eval_loader = None
    if config.evaluation.enabled:
        if not config.eval_data.root or not config.eval_data.ann_file:
            raise ValueError("Evaluation is enabled but eval_data.root or eval_data.ann_file is missing.")
        eval_dataset = CocoDetectionKD(
            root=config.eval_data.root,
            annFile=config.eval_data.ann_file,
        )
        eval_loader = build_detection_loader(eval_dataset, config.loader, shuffle=False)

    model = build_student(
        num_classes=config.student.num_classes,
        backbone_name=config.student.backbone_name,
        detector_weights=config.student.detector_weights,
        backbone_weights=config.student.backbone_weights,
        trainable_backbone_layers=config.student.trainable_backbone_layers,
    ).to(device)

    optimizer_name = config.optimizer.name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.optimizer.lr,
            betas=tuple(config.optimizer.betas),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

    scheduler = build_scheduler(optimizer, config.scheduler)

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, config.training.num_epochs + 1):
        loader = build_epoch_loader(train_dataset, config.loader, config.training, epoch)
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")
        print(f"Training samples this epoch: {len(loader.dataset)}")
        stats = {}
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            loss_dict = model(images, targets)
            loss_total = sum(loss_dict.values())
            loss_total.backward()
            optimizer.step()

            stats["loss_total"] = stats.get("loss_total", 0.0) + float(loss_total.detach())
            for key, value in loss_dict.items():
                stats[key] = stats.get(key, 0.0) + float(value.detach())

            pbar.set_postfix({"total": f"{loss_total.item():.3f}"})

        num_batches = max(len(loader), 1)
        stats = {key: value / num_batches for key, value in stats.items()}
        print("epoch_summary", json.dumps(stats, sort_keys=True))

        if eval_loader is not None and epoch % config.evaluation.interval == 0:
            eval_metrics = evaluate_detection_model(model=model, loader=eval_loader, device=device)
            print("eval_summary", json.dumps(eval_metrics, sort_keys=True))

        if scheduler is not None:
            scheduler.step()

    final_checkpoint = checkpoint_dir / f"{config.training.run}_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Model saved to {final_checkpoint}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.set)

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return

    train_vanilla_from_config(config)


if __name__ == "__main__":
    main()
