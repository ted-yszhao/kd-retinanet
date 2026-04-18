from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data.coco import CocoDetectionKD, detection_collate_fn
from src.evaluation.detection import evaluate_detection_model
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
    trainable_model = model.student if hasattr(model, "student") else model
    params = [p for p in trainable_model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config.lr,
            betas=tuple(optimizer_config.betas),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")


def build_scheduler(optimizer, scheduler_config):
    name = scheduler_config.name.lower()

    if name == "none":
        return None

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.t_max,
            eta_min=scheduler_config.eta_min,
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


def build_detection_loader(dataset, loader_config, shuffle=None):
    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle if shuffle is None else shuffle,
        num_workers=loader_config.num_workers,
        collate_fn=detection_collate_fn,
    )


def build_epoch_loader(train_dataset, loader_config, training_config, epoch: int):
    sample_ratio = training_config.sample_ratio
    if sample_ratio <= 0 or sample_ratio > 1:
        raise ValueError(f"training.sample_ratio must be in (0, 1], got {sample_ratio}.")

    if sample_ratio >= 1.0:
        return build_detection_loader(train_dataset, loader_config)

    sample_size = max(1, int(len(train_dataset) * sample_ratio))
    generator = torch.Generator()
    seed = training_config.sample_seed
    if training_config.resample_each_epoch:
        seed += epoch - 1
    generator.manual_seed(seed)

    indices = torch.randperm(len(train_dataset), generator=generator)[:sample_size].tolist()
    subset = Subset(train_dataset, indices)
    return build_detection_loader(subset, loader_config)


def build_training_components(config: ExperimentConfig):
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

    student = build_student(
        num_classes=config.student.num_classes,
        backbone_name=config.student.backbone_name,
        detector_weights=config.student.detector_weights,
        backbone_weights=config.student.backbone_weights,
        trainable_backbone_layers=config.student.trainable_backbone_layers,
    ).to(device)

    kd_enabled = (
        config.teacher is not None
        and config.kd is not None
        and config.kd.enabled
    )

    if kd_enabled:
        teacher = build_teacher(
            num_classes=config.teacher.num_classes,
            backbone_name=config.teacher.backbone_name,
            backbone_weights=config.teacher.backbone_weights,
            trainable_layers=config.teacher.trainable_layers,
            returned_layers=config.teacher.returned_layers,
            extra_blocks_in_channels=config.teacher.extra_blocks_in_channels,
            extra_blocks_out_channels=config.teacher.extra_blocks_out_channels,
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
    else:
        if config.evaluation.enabled and config.evaluation.branch == "teacher":
            raise ValueError("evaluation.branch='teacher' requires teacher/KD training to be enabled.")
        model = student

    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler)
    return model, train_dataset, eval_loader, optimizer, scheduler, device


def train(
    model,
    train_dataset,
    loader_config,
    training_config,
    optimizer,
    scheduler,
    device,
    num_epochs=3,
    run=None,
    log_dir="runs",
    checkpoint_dir="checkpoints",
    eval_loader=None,
    eval_branch="student",
    eval_metric="bbox_mAP",
    eval_interval=1,
):

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
        loader = build_epoch_loader(train_dataset, loader_config, training_config, epoch)
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Training samples this epoch: {len(loader.dataset)}")
        stats = {}
        # Batch loop
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for step, (images, targets) in enumerate(pbar, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            if "loss_total" not in losses:
                det_loss = sum(losses.values())
                losses = {
                    "loss_total": det_loss,
                    "loss_det": det_loss,
                    "loss_kd": det_loss.new_zeros(()),
                    **losses,
                }
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

        if eval_loader is not None and epoch % eval_interval == 0:
            eval_metrics = evaluate_detection_model(
                model=model,
                loader=eval_loader,
                device=device,
                branch=eval_branch,
            )
            metric_value = eval_metrics.get(eval_metric)
            metric_display = f"{metric_value:.4f}" if metric_value is not None else "n/a"
            summary_keys = [
                "bbox_mAP",
                "bbox_mAP_50",
                "bbox_mAP_75",
                "bbox_mAP_small",
                "bbox_mAP_medium",
                "bbox_mAP_large",
            ]
            summary = ", ".join(
                f"{key}={eval_metrics[key]:.4f}"
                for key in summary_keys
                if key in eval_metrics
            )
            print(f"Eval {eval_metric}: {metric_display}")
            print(f"Eval metrics: {summary}")
            for key, value in eval_metrics.items():
                writer.add_scalar(f"eval/{key}", value, epoch)
            writer.flush()

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)


    final_checkpoint = checkpoint_path / f"{run_name}_final.pth"
    trainable_model = model.student if hasattr(model, "student") else model
    torch.save(trainable_model.state_dict(), final_checkpoint)
    print(f"Model saved to {final_checkpoint}")
    writer.close()


def train_from_config(config: ExperimentConfig):
    model, train_dataset, eval_loader, optimizer, scheduler, device = build_training_components(config)
    train(
        model=model,
        train_dataset=train_dataset,
        loader_config=config.loader,
        training_config=config.training,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config.training.num_epochs,
        run=config.training.run,
        log_dir=config.training.log_dir,
        checkpoint_dir=config.training.checkpoint_dir,
        eval_loader=eval_loader,
        eval_branch=config.evaluation.branch,
        eval_metric=config.evaluation.metric,
        eval_interval=config.evaluation.interval,
    )
