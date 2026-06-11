"""FastAPI app for optical diffraction image diagnosis."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile

from optidiag.api.schemas import BatchDiagnosisResponse, DiagnosisResponse, HealthResponse
from optidiag.api.utils import ensure_supported_experiment, load_request_image, pil_to_array
from optidiag.analysis.diffraction_metrics import compute_diffraction_metrics
from optidiag.models.inference import ModelInference
from optidiag.reasoning.rules import diagnose_from_metrics


app = FastAPI(
    title="OptiDiag Agent API",
    description=(
        "Optical diffraction experiment image diagnosis API. "
        "POST /analyze supports multipart file upload and optional image_url; file takes precedence."
    ),
    version="0.1.0",
)

model_inference = ModelInference()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and fallback mode."""
    return HealthResponse(model_available=model_inference.available)


def _analyze_array(image: object) -> DiagnosisResponse:
    arr = pil_to_array(image)
    model_prediction = model_inference.predict(arr)
    image_type = "unknown_or_estimated"
    confidence = 0.0
    fallback = "rule_based"
    if model_prediction:
        image_type = str(model_prediction["image_type"])
        confidence = float(model_prediction["confidence"])
        fallback = "model_with_rule_based"
    elif model_inference.available and getattr(model_inference, "last_prediction_failed", False):
        fallback = "rule_based_after_model_failure"

    metrics = compute_diffraction_metrics(arr)
    response = diagnose_from_metrics(metrics, image_type=image_type, confidence=confidence)
    if model_prediction and model_prediction.get("issues"):
        # Stage-2 hook: trained model issues can replace rule issues while the report stays Chinese.
        response["issues"] = model_prediction["issues"]
    response["model_available"] = bool(model_inference.available)
    response["fallback"] = fallback
    return DiagnosisResponse(**response)


@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze(
    file: Optional[UploadFile] = File(default=None, description="Image file. Takes precedence over image_url."),
    image_url: Optional[str] = Form(default=None, description="Optional HTTP/HTTPS image URL when no file is uploaded."),
    experiment_type: str = Form(default="diffraction", description="Only diffraction is supported in the MVP."),
) -> DiagnosisResponse:
    """Analyze one diffraction experiment image."""
    ensure_supported_experiment(experiment_type)
    image = await load_request_image(file=file, image_url=image_url)
    return _analyze_array(image)


@app.post("/analyze_batch", response_model=BatchDiagnosisResponse)
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Multiple image files."),
    experiment_type: str = Form(default="diffraction"),
) -> BatchDiagnosisResponse:
    """Analyze multiple uploaded files. This endpoint is optional for the MVP."""
    ensure_supported_experiment(experiment_type)
    results = []
    errors = []
    for file in files:
        try:
            image = await load_request_image(file=file, image_url=None)
            results.append(_analyze_array(image))
        except Exception as exc:  # noqa: BLE001
            errors.append({"file": file.filename, "error": str(exc)})
    return BatchDiagnosisResponse(results=results, errors=errors)
