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

