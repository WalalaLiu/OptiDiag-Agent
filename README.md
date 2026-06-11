# OptiDiag Agent

OptiDiag Agent is a minimal runnable MVP for an optical diffraction experiment image diagnosis agent plugin. It is based on the legacy `prompt/` simulation and inference ideas, but the task is upgraded from simple aperture classification to experiment-quality diagnosis.

The MVP answers:

- What diffraction image type is present or estimated?
- What acquisition problems may exist?
- How severe are the problems?
- Which image metrics support the diagnosis?
- What should the experimenter adjust next?

Supported image types:

- `single_slit`
- `double_slit`
- `triple_slit`
- `rectangle_aperture`
- `circular_aperture`
- `double_circular_aperture`

Supported issue labels:

- `over_exposure`
- `under_exposure`
- `gaussian_noise`
- `poisson_noise`
- `dark_noise`
- `background_gradient`
- `blur_defocus`
- `misalignment`
- `rotation_tilt`
- `cropping_incomplete`
- `low_contrast`

## Why This Is Not Simple Classification

The legacy demo classifies aperture categories. This project also computes interpretable metrics and diagnoses experimental acquisition errors. When no neural-network checkpoint exists, `/analyze` still works through rule-based fallback logic.

Metrics include saturation ratio, dark-pixel ratio, center offset, fringe visibility, Laplacian variance, background uniformity, FFT peak strength, and central lobe width. These values are returned in JSON so an AI agent can explain its conclusion.

## Install

```bash
cd OptiDiag-Agent
python3 -m pip install -r requirements.txt
```

Optional training dependencies:

```bash
python3 -m pip install torch torchvision
```

GPU is not required for the MVP. GPU is only useful for later multi-task training.

## Generate Smoke-Test Data

```bash
python scripts/generate_dataset.py --num 20 --out data/simulated/smoke --image-size 128 --seed 42
```

Full MVP dataset example:

```bash
python scripts/generate_dataset.py --num 200 --out data/simulated/diffraction_mvp --image-size 256 --seed 42
```

Generated data is ignored by Git.

## Run API

```bash
python scripts/run_api.py
```

The service starts at:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Analyze With File Upload

`file` is the preferred input. If both `file` and `image_url` are provided, `file` takes precedence.

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@data/simulated/smoke/images/00000_single_slit_over_exposure.png" \
  -F "experiment_type=diffraction"
```

## Analyze With image_url

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image_url=https://example.com/diffraction.png" \
  -F "experiment_type=diffraction"
```

## Example Response

```json
{
  "image_type": "unknown_or_estimated",
  "confidence": 0.0,
  "issues": [
    {"type": "over_exposure", "severity": "medium", "score": 0.62}
  ],
  "metrics": {
    "saturation_ratio": 0.08,
    "center_offset_px": [14.0, -6.0],
    "fringe_visibility": 0.71,
    "laplacian_variance": 0.0043,
    "snr_estimate": 1.85
  },
  "diagnosis": "图像疑似存在中等程度过曝。",
  "possible_causes": ["相机曝光时间或光源功率偏高"],
  "suggestions": ["适当降低曝光时间或光源强度，避免中央主极大饱和"],
  "need_reacquire": true,
  "model_available": false,
  "fallback": "rule_based"
}
```

## OpenAPI

FastAPI exposes:

```text
GET /openapi.json
```

Export a static copy:

```bash
python scripts/export_openapi.py --out docs/openapi.json
```

Use this schema when configuring an agent plugin. The important endpoint is `POST /analyze`; it supports multipart `file`, optional `image_url`, and `experiment_type=diffraction`.

## Training

Stage 1 does not require training. A model skeleton is provided for stage 2:

```bash
python scripts/train.py --data data/simulated/diffraction_mvp --epochs 5
```

If `models/checkpoints/best_model.pt` is absent, the API automatically uses rule-based fallback and does not crash.

## Report Screenshot Suggestions

- `outputs/demo_cases/summary.csv`
- `outputs/demo_cases/visualizations/*_report.png`
- Generated dataset folder with `images/`, `metadata/`, and `labels.csv`
- One JSON metadata example showing issue labels and metrics
- `/health` response
- `/analyze` response for over-exposure, blur, and misalignment examples
- OpenAPI schema page or exported `docs/openapi.json`
- Rule-based diagnosis explanation and Chinese suggestions
