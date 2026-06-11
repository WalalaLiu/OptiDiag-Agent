# Project Plan

## Stage 1: MVP

Goal: make the project runnable and agent-plugin ready without requiring a trained model.

Deliverables:

- New `OptiDiag-Agent/` code structure
- Six-class diffraction simulation
- Explicit issue labels and `labels.csv`
- Explainable image metrics
- Rule-based fallback diagnosis
- FastAPI `/health`, `/analyze`, `/analyze_batch`, `/openapi.json`
- OpenAPI export script
- Smoke tests and smoke dataset generation

Success criteria:

- `python scripts/generate_dataset.py --num 20 --out data/simulated/smoke --image-size 128 --seed 42` works
- `/analyze` returns issue scores, metrics, Chinese diagnosis, causes, and suggestions
- Service works without GPU or model weights

## Stage 2: Multi-Task Training

Goal: train a model that complements or replaces rule-only diagnosis.

Planned model:

- Shared CNN or ResNet18 backbone
- Head 1: aperture/image type classification
- Head 2: issue multi-label classification
- Head 3: quality score regression

Needed assets:

- Larger simulated dataset with balanced issue labels
- Small real-image validation set from the lab
- Confusion matrix, multi-label F1, regression MAE
- Failure-case analysis

## Stage 3: Agent Integration and Report Materials

Goal: connect the API to the course agent platform and collect report evidence.

Tasks:

- Configure OpenAPI tool in the agent platform
- Prepare system prompt for interpreting JSON results
- Capture screenshots of upload, plugin call, and diagnosis answer
- Run 3-5 example scenarios: overexposure, misalignment, blur, background gradient, low contrast
- Collect user feedback screenshots or short comments

## 进入深度学习训练前的检查清单

- 已确认 `pytest`、smoke 数据生成、OpenAPI 导出都能通过。
- 已生成 `outputs/demo_cases/` 演示材料，并检查可视化图片可用于报告截图。
- 已确认 `/analyze` 返回字段稳定，包括 `model_available` 和 `fallback`。
- 已准备至少 200-1000 张平衡的仿真训练图像，覆盖 6 类衍射图和 11 类误差。
- 已准备少量真实实验图像，并按 `data/real/README.md` 标注。
- 已确认真实图像不提交到 GitHub，只保存目录说明和标注模板。
- 已检查 `data/simulated/`、`outputs/`、`runs/`、`.venv/` 和模型权重均被 `.gitignore` 忽略。
- 已确定训练目标：图像类型分类、误差多标签分类、质量分数回归。
- 已决定训练评估指标：分类准确率、多标签 F1、质量分数 MAE、典型失败案例。
- 已确认是否使用 GPU 服务器；若使用，记录 Python、PyTorch、CUDA 版本。
