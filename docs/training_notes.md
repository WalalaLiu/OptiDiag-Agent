# Training Notes

Phase 2A adds a small multi-task training path. The goal is not high final accuracy yet; the goal is to verify that labels, model outputs, checkpoints, and evaluation all work.

## 生成仿真训练集

Small CPU smoke dataset:

```bash
python scripts/generate_dataset.py --num 600 --out data/simulated/train_smoke --image-size 128 --seed 2026
```

Dataset layout:

```text
data/simulated/train_smoke/
  images/
  metadata/
  labels.csv
```

`data/simulated/` is ignored by Git.

## labels.csv 字段含义

- `image_path`: image path relative to dataset directory.
- `json_path`: metadata path relative to dataset directory.
- `image_type`: one of six diffraction image classes.
- `primary_issue`: main simulated acquisition issue.
- `issues`: JSON list of active issues.
- `issue_scores`: JSON dictionary with all issue scores.
- `severity`: `low`, `medium`, or `high`.
- `noise_type`: active noise type when applicable.
- `noise_level`: synthetic noise strength.
- `blur_sigma`: Gaussian blur radius for defocus.
- `rotation_deg`: synthetic rotation angle.
- `translation_x`, `translation_y`: synthetic center shift in pixels.
- `exposure_factor`: exposure scaling factor.
- `background_gradient_strength`: synthetic gradient strength.
- `contrast`, `snr_estimate`, `saturation_ratio`, `fringe_visibility`: image metrics.
- `center_offset_x`, `center_offset_y`: center-of-mass offset.
- `quality_score`: regression target in `[0, 1]`.
- `suggestion_label`: coarse operation suggestion label.

## 数据均衡检查

```bash
python scripts/check_dataset_balance.py --labels data/simulated/train_smoke/labels.csv
```

The checker prints:

- six-class `image_type` counts
- active `issue` counts
- `primary_issue` counts
- `severity` counts
- imbalance warnings when max/min ratios are too large

## 小规模训练命令

```bash
python scripts/train.py \
  --data-dir data/simulated/train_smoke \
  --epochs 2 \
  --batch-size 16 \
  --device cpu
```

The model has three heads:

- `image_type_logits`: six-class image type classification
- `issue_logits`: multi-label issue classification
- `quality_score`: regression target for overall image quality

Checkpoint:

```text
models/checkpoints/best_model.pt
```

This checkpoint is ignored by Git.

## 评估命令

```bash
python scripts/evaluate.py \
  --data-dir data/simulated/train_smoke \
  --checkpoint models/checkpoints/best_model.pt \
  --device cpu
```

Evaluation metrics:

- `image_type_accuracy`: six-class image type accuracy
- `issue_micro_f1`: multi-label issue micro F1
- `issue_macro_f1`: average issue F1 across issue labels
- `primary_issue_top1_accuracy`: whether the highest issue probability matches `primary_issue`
- `quality_mae`: mean absolute error for `quality_score`

## 什么时候需要 GPU

CPU is enough for Phase 2A smoke tests with 300-600 images and 128 px inputs.

GPU becomes useful when:

- training more than several thousand images
- using 224 px or larger inputs
- replacing the lightweight CNN with ResNet18/34
- running many hyperparameter sweeps
- mixing real and synthetic data with heavier augmentation

The API does not require GPU. If `models/checkpoints/best_model.pt` is missing or invalid, `/analyze` still falls back to `rule_based`.

