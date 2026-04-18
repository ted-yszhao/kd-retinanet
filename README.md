# KD-RetinaNet

KD-RetinaNet is a RetinaNet knowledge distillation training project built around a central config file and a small CLI. The current setup supports configurable teacher and student models, COCO evaluation through `pycocotools`, TensorBoard logging, and per-epoch subset sampling for cheaper experiments.

## Setup

### 1. Python

Use Python `3.10+`.

### 2. Install dependencies

At minimum, training currently expects:

- `torch`
- `torchvision`
- `kornia`
- `tensorboard`
- `pycocotools`
- `tqdm`

Example install:

```bash
pip install torch torchvision kornia tensorboard pycocotools tqdm
```

The repo also includes a minimal [pyproject.toml](/root/kd-retinanet/pyproject.toml), but it does not yet list the full training stack, so the manual install above is the safest setup path right now.

### 3. Prepare COCO data

The default config expects this layout:

```text
datasets/
  coco/
    annotations/
      instances_train2017.json
      instances_val2017.json
    val2017/

../data/
  train2017/
```

With the current default config:

- train images: `../data/train2017`
- train annotations: `datasets/coco/annotations/instances_train2017.json`
- val images: `datasets/coco/val2017`
- val annotations: `datasets/coco/annotations/instances_val2017.json`

If your paths are different, update the config file instead of editing code.

## Run Training

The main entrypoint is [main.py](/root/kd-retinanet/main.py).

Print the resolved config:

```bash
python main.py --print-config
```

Train with the default config:

```bash
python main.py
```

Train with a different config file:

```bash
python main.py --config configs/resnet101_teacher_resnet50_student.json
```

Train a vanilla RetinaNet detector with a `resnet18` backbone:

```bash
python main.py --config configs/vanilla_retinanet_resnet18.json
```

Override values from the CLI:

```bash
python main.py \
  --set training.num_epochs=1 \
  --set optimizer.name="adamw" \
  --set optimizer.lr=1e-4 \
  --set training.sample_ratio=0.1
```

## Config File

The default experiment config is [configs/resnet101_teacher_resnet50_student.json](/root/kd-retinanet/configs/resnet101_teacher_resnet50_student.json).

For standalone detector training without knowledge distillation, use [configs/vanilla_retinanet_resnet18.json](/root/kd-retinanet/configs/vanilla_retinanet_resnet18.json). That config now runs directly through [main.py](/root/kd-retinanet/main.py) with `teacher: null` and `kd.enabled: false`.

It is organized into these sections:

- `data`: training dataset paths
- `eval_data`: validation dataset paths
- `loader`: batch size, shuffle, worker count
- `teacher`: teacher RetinaNet / backbone settings
- `student`: student RetinaNet / backbone settings
- `kd`: KD loss settings
- `optimizer`: optimizer selection and hyperparameters
- `scheduler`: learning rate scheduler settings
- `training`: epochs, run name, subset sampling, output directories
- `evaluation`: evaluation frequency and selected metric

### Important fields

`data`

- `root`: training image directory
- `ann_file`: training annotation json

`eval_data`

- `root`: validation image directory
- `ann_file`: validation annotation json

`loader`

- `batch_size`
- `shuffle`
- `num_workers`

`optimizer`

- `name`: `sgd` or `adamw`
- `lr`
- `momentum`: used by `sgd`
- `betas`: used by `adamw`
- `eps`: used by `adamw`
- `weight_decay`

`scheduler`

- `name`: `none`, `step`, or `cosine`
- `step_size`
- `gamma`
- `t_max`
- `eta_min`

`training`

- `num_epochs`
- `sample_ratio`: fraction of the training dataset used each epoch
- `resample_each_epoch`: if `true`, draw a new random subset each epoch
- `sample_seed`
- `run`: TensorBoard run name and checkpoint prefix
- `log_dir`
- `checkpoint_dir`

`evaluation`

- `enabled`
- `branch`: usually `student`
- `interval`: run evaluation every N epochs
- `metric`: currently `bbox_mAP`

`teacher`

- set to `null` to train only the student model with no teacher branch

`kd`

- `enabled`: set to `false` to skip KD loss and train only with detector losses

`student`

- `backbone_name`: `resnet50` or `resnet18`
- `detector_weights`: only supported with `resnet50`
- `backbone_weights`
- `trainable_backbone_layers`

## Common Recipes

Train on full data each epoch:

```bash
python main.py --set training.sample_ratio=1.0
```

Train on half of COCO each epoch with a new random subset:

```bash
python main.py \
  --set training.sample_ratio=0.5 \
  --set training.resample_each_epoch=true
```

Use SGD:

```bash
python main.py \
  --set optimizer.name="sgd" \
  --set optimizer.lr=0.001 \
  --set scheduler.name="step"
```

Use AdamW:

```bash
python main.py \
  --set optimizer.name="adamw" \
  --set optimizer.lr=1e-4 \
  --set scheduler.name="cosine"
```

Run a quick smoke train:

```bash
python main.py \
  --set training.num_epochs=1 \
  --set training.sample_ratio=0.001 \
  --set loader.num_workers=0 \
  --set evaluation.enabled=false \
  --set training.run="smoke_train"
```

Use a `resnet18` student:

```bash
python main.py \
  --set student.backbone_name="resnet18" \
  --set student.detector_weights=null \
  --set training.run="resnet101_teacher_resnet18_student"
```

Disable teacher/KD and train only the student from the main entrypoint:

```bash
python main.py \
  --set teacher=null \
  --set kd.enabled=false \
  --set training.run="student_only"
```

## Evaluation And TensorBoard

Evaluation uses off-the-shelf `pycocotools.COCOeval` through [src/evaluation/detection.py](/root/kd-retinanet/src/evaluation/detection.py).

When evaluation is enabled, the training loop logs:

- `eval/bbox_mAP`
- `eval/bbox_mAP_50`
- `eval/bbox_mAP_75`
- `eval/bbox_mAP_small`
- `eval/bbox_mAP_medium`
- `eval/bbox_mAP_large`
- `eval/num_images`

Training logs are written under `runs/<run_name>`.

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

Checkpoints are written to `checkpoints/<run_name>_final.pth`.
