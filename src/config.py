from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    root: str
    ann_file: str


@dataclass
class EvalDataConfig:
    root: str | None = None
    ann_file: str | None = None


@dataclass
class LoaderConfig:
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 2


@dataclass
class TeacherConfig:
    num_classes: int = 91
    backbone_name: str = "resnet101"
    backbone_weights: str | None = "DEFAULT"
    trainable_layers: int = 3
    returned_layers: list[int] = field(default_factory=lambda: [2, 3, 4])
    extra_blocks_in_channels: int = 256
    extra_blocks_out_channels: int = 256


@dataclass
class StudentConfig:
    num_classes: int = 91
    backbone_name: str = "resnet50"
    detector_weights: str | None = None
    backbone_weights: str | None = "DEFAULT"
    trainable_backbone_layers: int = 3


@dataclass
class KDConfig:
    enabled: bool = True
    keys: list[str] = field(default_factory=lambda: ["2"])
    weights: list[float] = field(default_factory=lambda: [1.0])
    window_size: int = 7
    kd_weight: float = 0.5


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.001
    momentum: float = 0.9
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 1e-4


@dataclass
class SchedulerConfig:
    name: str = "none"
    step_size: int = 8
    gamma: float = 0.1
    t_max: int = 10
    eta_min: float = 0.0


@dataclass
class TrainingConfig:
    device: str = "auto"
    num_epochs: int = 10
    sample_ratio: float = 1.0
    resample_each_epoch: bool = True
    sample_seed: int = 42
    run: str = "resnet101_teacher_resnet50_student"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvaluationConfig:
    enabled: bool = False
    branch: str = "student"
    interval: int = 1
    metric: str = "bbox_mAP"


@dataclass
class ExperimentConfig:
    data: DataConfig
    eval_data: EvalDataConfig = field(default_factory=EvalDataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    teacher: TeacherConfig | None = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    kd: KDConfig | None = field(default_factory=KDConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _parse_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def apply_overrides(config_dict: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = json.loads(json.dumps(config_dict))
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        value = _parse_override_value(raw_value)

        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return result


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    teacher_payload = payload.get("teacher", {})
    kd_payload = payload.get("kd", {})
    return ExperimentConfig(
        data=DataConfig(**payload["data"]),
        eval_data=EvalDataConfig(**payload.get("eval_data", {})),
        loader=LoaderConfig(**payload.get("loader", {})),
        teacher=None if teacher_payload is None else TeacherConfig(**teacher_payload),
        student=StudentConfig(**payload.get("student", {})),
        kd=None if kd_payload is None else KDConfig(**kd_payload),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        scheduler=SchedulerConfig(**payload.get("scheduler", {})),
        training=TrainingConfig(**payload.get("training", {})),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    payload = apply_overrides(payload, overrides)
    return experiment_config_from_dict(payload)


def save_config(path: str | Path, config: ExperimentConfig) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
        f.write("\n")
