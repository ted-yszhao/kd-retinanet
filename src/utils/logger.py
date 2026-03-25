import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class TrainLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.tb_dir = self.run_dir / "tb"
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.csv_path = self.run_dir / "scalars.csv"
        self.meta_path = self.run_dir / "meta.json"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "global_step",
                "epoch",
                "iter_in_epoch",
                "lr",
                "loss_total",
                "loss_cls",
                "loss_box",
                "loss_kd",
                "grad_norm",
                "param_norm",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def log_iter(self, *, global_step, epoch, iter_in_epoch, lr,
                 loss_total, loss_cls=None, loss_box=None, loss_kd=None,
                 grad_norm=None, param_norm=None):

        row = {
            "global_step": global_step,
            "epoch": epoch,
            "iter_in_epoch": iter_in_epoch,
            "lr": lr,
            "loss_total": loss_total,
            "loss_cls": "" if loss_cls is None else loss_cls,
            "loss_box": "" if loss_box is None else loss_box,
            "loss_kd": "" if loss_kd is None else loss_kd,
            "grad_norm": "" if grad_norm is None else grad_norm,
            "param_norm": "" if param_norm is None else param_norm,
        }
        self.csv_writer.writerow(row)

        self.writer.add_scalar("train/loss_total", loss_total, global_step)
        self.writer.add_scalar("train/loss_cls", loss_cls or 0.0, global_step)
        self.writer.add_scalar("train/loss_box", loss_box or 0.0, global_step)
        if loss_kd is not None:
            self.writer.add_scalar("train/loss_kd", loss_kd, global_step)
        self.writer.add_scalar("train/lr", lr, global_step)

        if grad_norm is not None:
            self.writer.add_scalar("train/grad_norm", grad_norm, global_step)
        if param_norm is not None:
            self.writer.add_scalar("train/param_norm", param_norm, global_step)

        # cheap flush cadence; adjust if you want
        if global_step % 50 == 0:
            self.csv_file.flush()
            self.writer.flush()

    def save_checkpoint(self, name, model, optimizer, scheduler, epoch, global_step, extra=None):
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "extra": extra or {},
        }
        path = self.ckpt_dir / name
        torch.save(ckpt, path)
        return path

    def save_metadata(self, metadata: dict):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def close(self):
        self.csv_file.flush()
        self.csv_file.close()
        self.writer.flush()
        self.writer.close()



def get_param_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        total += p.detach().data.norm(2).item() ** 2
    return total ** 0.5


def get_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5