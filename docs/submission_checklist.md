# Submission Checklist

Use this checklist before packaging the course project.

## Required Materials

- Project report
- Source code
- Dataset description
- Model training notes
- API documentation
- User guide
- Demo outputs
- Server training result summary

## Source Code

Include:

- `src/optidiag/`
- `scripts/`
- `configs/`
- `docs/`
- `tests/`
- `README.md`
- `requirements.txt`
- `pyproject.toml`

Keep:

- `prompt/` legacy reference code
- `legacy/README.md`

Do not include:

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- generated datasets
- model checkpoints
- local server logs

## Dataset Materials

Include:

- `data/README.md`
- `data/real/README.md`
- description of simulated data generation command
- labels field explanation
- representative screenshots or small non-sensitive examples if allowed

Do not commit:

```text
data/simulated/
data/real/raw/
```

## Model Training Materials

Include:

- `docs/training_notes.md`
- `docs/model_evaluation_notes.md`
- training command
- evaluation command
- threshold calibration result summary
- final metrics table

Do not commit:

```text
models/checkpoints/*.pt
models/checkpoints/*.pth
runs/
```

## API and Plugin Materials

Include:

- `docs/openapi_notes.md`
- `docs/api_examples.md`
- `docs/final_demo_guide.md`
- `docs/openapi.json`
- screenshots of `/health` and `/analyze`

## Demo Materials

Generate locally or on server:

```bash
python scripts/make_demo_cases.py
python scripts/visualize_demo_cases.py
```

Use screenshots from:

```text
outputs/demo_cases/summary.csv
outputs/demo_cases/visualizations/
```

Do not commit:

```text
outputs/
```

## Final Verification

Run:

```bash
pytest
python scripts/generate_dataset.py --num 20 --out data/simulated/smoke --image-size 128 --seed 42
python scripts/make_demo_cases.py
python scripts/export_openapi.py
```

On the training server, also run:

```bash
python scripts/evaluate.py --data-dir data/simulated/train_smoke --checkpoint models/checkpoints/best_model.pt --device cuda
python scripts/inspect_predictions.py --data-dir data/simulated/train_smoke --checkpoint models/checkpoints/best_model.pt --device cuda
```

