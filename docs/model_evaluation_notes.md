# Model Evaluation Notes

## Phase 2A Baseline

Small training validation showed that the model plumbing worked, but issue prediction needed better calibration.

Phase 2A baseline on 600 simulated images, 10 epochs, RTX 3090:

```text
image_type_accuracy=0.5517
issue_micro_f1=0.0
issue_macro_f1=0.0
primary_issue_top1_accuracy=0.1467
quality_mae=0.0958
threshold=0.5
```

Interpretation:

- The image-type head had started learning.
- The quality regression head was usable for smoke validation.
- The issue multi-label head produced probabilities below the fixed threshold, so F1 was zero at `threshold=0.5`.

## Phase 2B v1 Result

After adding issue `pos_weight`, issue loss weighting, and better diagnostics:

```text
dataset_size=3000
image_type_accuracy=0.6100
issue_micro_f1=0.4800
issue_macro_f1=0.4964
primary_issue_top1_accuracy=0.4553
quality_mae=0.0749
best_issue_threshold=0.5
```

Interpretation:

- The issue head is now learning meaningful signals.
- Some labels, especially `misalignment`, `rotation_tilt`, and `cropping_incomplete`, can have high recall but low precision.
- Since the best threshold was at the upper edge of the old sweep, the sweep range needed to extend above 0.5.

## Why Multi-Label Tasks Need Threshold Calibration

The issue head is multi-label: one image can have multiple active issues. The model outputs one probability per issue, but the threshold that converts probabilities into labels is not naturally fixed at 0.5.

Reasons:

- Some issue labels are easier and produce high probabilities.
- Some issue labels are visually correlated, such as misalignment and cropping.
- Positive/negative class balance differs by issue.
- A single global threshold can be too permissive for one issue and too strict for another.

Phase 2C evaluates thresholds:

```text
0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
```

It reports both global thresholds and per-issue best thresholds.

## Precision and Recall Trade-Off

- Higher threshold usually increases precision and lowers recall.
- Lower threshold usually increases recall and lowers precision.
- For classroom experiment guidance, false positives are not always equally bad: warning about a possible issue can still be useful if the explanation is clear.
- For automatic scoring or grading, higher precision may matter more.

Suggested usage:

- Use `best_micro_threshold` when overall issue detection is the priority.
- Use `best_macro_threshold` when all issue classes should contribute equally.
- Use per-class thresholds when some labels are systematically over-predicted or under-predicted.

## Plugin Strategy

The final plugin should not rely entirely on the neural network.

Recommended runtime behavior:

1. Use the model for `image_type` and issue probability estimates when a checkpoint is available.
2. Use calibrated thresholds to turn issue probabilities into candidate issues.
3. Keep rule-based metrics and explanations in the response.
4. If the model is missing, invalid, or fails during inference, return the `rule_based` fallback.

This hybrid approach is safer for teaching because the agent can still explain its reasoning through metrics such as saturation ratio, center offset, fringe visibility, background gradient, and Laplacian variance.

