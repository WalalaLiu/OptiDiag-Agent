# API Examples

## Health

```bash
curl http://127.0.0.1:8000/health
```

Example:

```json
{
  "status": "ok",
  "service": "optidiag-agent",
  "model_available": false,
  "fallback": "rule_based"
}
```

## Analyze: File Upload

`file` is preferred and takes precedence over `image_url`.

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@data/simulated/smoke/images/00000_single_slit_over_exposure.png" \
  -F "experiment_type=diffraction"
```

## Analyze: image_url

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image_url=https://example.com/diffraction.png" \
  -F "experiment_type=diffraction"
```

## Response Shape

```json
{
  "image_type": "unknown_or_estimated",
  "confidence": 0.0,
  "issues": [
    {"type": "over_exposure", "severity": "medium", "score": 0.62},
    {"type": "misalignment", "severity": "low", "score": 0.31}
  ],
  "metrics": {
    "saturation_ratio": 0.08,
    "center_offset_px": [14.0, -6.0],
    "fringe_visibility": 0.71,
    "laplacian_variance": 0.0043,
    "snr_estimate": 1.85
  },
  "diagnosis": "图像疑似存在中等程度过曝和轻微光轴偏移。",
  "possible_causes": [
    "相机曝光时间或光源功率偏高",
    "孔径、透镜和相机中心未完全共轴"
  ],
  "suggestions": [
    "适当降低曝光时间或光源强度，避免中央主极大饱和",
    "微调孔径、透镜和相机中心，使主极大回到图像中心"
  ],
  "need_reacquire": true
}
```

