from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.kd.ssim_kd import MultiScaleSSIMKD
from src.models.distiller import RetinaNetDistiller
from src.models.retinanet_builder import build_teacher, build_student

from torch.utils.tensorboard.writer import SummaryWriter


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_optimizer(model, optimizer_config):
    name = optimizer_config.name.lower()
    params = [p for p in model.student.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")


def build_training_components(config: ExperimentConfig):
    device = resolve_device(config.training.device)

    dataset = CocoDetectionKD(
        root=config.data.root,
        annFile=config.data.ann_file,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        shuffle=config.loader.shuffle,
        num_workers=config.loader.num_workers,
        collate_fn=detection_collate_fn,
    )

    teacher = build_teacher(
        num_classes=config.teacher.num_classes,
        backbone_name=config.teacher.backbone_name,
        backbone_weights=config.teacher.backbone_weights,
        trainable_layers=config.teacher.trainable_layers,
        returned_layers=config.teacher.returned_layers,
        extra_blocks_in_channels=config.teacher.extra_blocks_in_channels,
        extra_blocks_out_channels=config.teacher.extra_blocks_out_channels,
    ).to(device)

    student = build_student(
        num_classes=config.student.num_classes,
        detector_weights=config.student.detector_weights,
        backbone_weights=config.student.backbone_weights,
        trainable_backbone_layers=config.student.trainable_backbone_layers,
    ).to(device)

    kd_loss = MultiScaleSSIMKD(
        keys=tuple(config.kd.keys),
        weights=tuple(config.kd.weights),
        window_size=config.kd.window_size,
    ).to(device)

    model = RetinaNetDistiller(
        teacher=teacher,
        student=student,
        kd_loss=kd_loss,
        kd_weight=config.kd.kd_weight,
    ).to(device)

    optimizer = build_optimizer(model, config.optimizer)
    return model, loader, optimizer, device


def train(model, loader, optimizer, device, num_epochs=3, run=None, log_dir="runs", checkpoint_dir="checkpoints"):

    run_name = run or "experiment"
    log_path = Path(log_dir) / run_name
    checkpoint_path = Path(checkpoint_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_path))
    global_step = 0
    model.train()

    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        stats = {}
        # Batch loop
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for step, (images, targets) in enumerate(pbar, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            losses["loss_total"].backward()
            optimizer.step()
            global_step += 1

            for k, v in losses.items():
                stats[k] = stats.get(k, 0.0) + float(v.detach())

            # Logging to TensorBoard
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

            # update tqdm display
            pbar.set_postfix({
                "total": f"{losses['loss_total'].item():.3f}",
                "det": f"{losses['loss_det'].item():.3f}",
                "kd": f"{losses['loss_kd'].item():.3f}",
            })

        # average stats
        num_batches = max(len(loader), 1)
        stats = {k: v / num_batches for k, v in stats.items()}


    final_checkpoint = checkpoint_path / f"{run_name}_final.pth"
    torch.save(model.student.state_dict(), final_checkpoint)
    print(f"Model saved to {final_checkpoint}")
    writer.close()


def train_from_config(config: ExperimentConfig):
    model, loader, optimizer, device = build_training_components(config)
    train(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.training.num_epochs,
        run=config.training.run,
        log_dir=config.training.log_dir,
        checkpoint_dir=config.training.checkpoint_dir,
    )
