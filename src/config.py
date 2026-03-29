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
    detector_weights: str | None = None
    backbone_weights: str | None = "DEFAULT"
    trainable_backbone_layers: int = 3


@dataclass
class KDConfig:
    keys: list[str] = field(default_factory=lambda: ["2"])
    weights: list[float] = field(default_factory=lambda: [1.0])
    window_size: int = 7
    kd_weight: float = 0.5


@dataclass
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    device: str = "auto"
    num_epochs: int = 10
    run: str = "resnet101_teacher_resnet50_student"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"


@dataclass
class ExperimentConfig:
    data: DataConfig
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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
    return ExperimentConfig(
        data=DataConfig(**payload["data"]),
        loader=LoaderConfig(**payload.get("loader", {})),
        teacher=TeacherConfig(**payload.get("teacher", {})),
        student=StudentConfig(**payload.get("student", {})),
        kd=KDConfig(**payload.get("kd", {})),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        training=TrainingConfig(**payload.get("training", {})),
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
