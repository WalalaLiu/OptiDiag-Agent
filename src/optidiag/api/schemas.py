"""Pydantic schemas for the OptiDiag HTTP API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IssueItem(BaseModel):
    """One detected issue with severity and score."""

    type: str = Field(..., examples=["over_exposure"])
    severity: str = Field(..., examples=["medium"])
    score: float = Field(..., ge=0.0, le=1.0, examples=[0.62])


class DiagnosisResponse(BaseModel):
    """Agent-facing diagnosis response."""

    image_type: str = Field(..., examples=["double_slit"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.0])
    issues: List[IssueItem]
    metrics: Dict[str, Any]
    diagnosis: str
    possible_causes: List[str]
    suggestions: List[str]
    need_reacquire: bool


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = "ok"
    service: str = "optidiag-agent"
    model_available: bool = False
    fallback: str = "rule_based"


class BatchDiagnosisResponse(BaseModel):
    """Batch diagnosis response."""

    results: List[DiagnosisResponse]
    errors: List[Dict[str, Optional[str]]] = []

