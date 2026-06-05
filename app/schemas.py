"""
app/schemas.py — Pydantic Request & Response Models
====================================================
CONCEPT: What is Pydantic?
Pydantic validates incoming data automatically. If a client sends
{"text": 123} instead of {"text": "some string"}, Pydantic rejects
it BEFORE your prediction code even runs — protecting the API.

Think of it as a "type-safe contract" between the client and your API.
"""
from pydantic import BaseModel, Field
from typing import Literal, List


class PredictRequest(BaseModel):
    """What the client sends TO the API."""
    text: str = Field(
        ...,                        # "..." means this field is REQUIRED
        min_length=1,
        max_length=5000,
        description="Conversational text to analyze (chat message, call transcript, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "If you don't comply with my demands, you will regret it."
            }
        }
    }


class ScoreDetail(BaseModel):
    """Confidence score for a single category."""
    label: str
    confidence: float


class PredictResponse(BaseModel):
    """What the API sends BACK to the client."""
    input_text: str
    prediction: str                                             # Top predicted label
    confidence: float                                           # Confidence for top label (0–1)
    is_threat: bool                                             # True if threatening or harassment
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]               # Human-readable risk
    all_scores: List[ScoreDetail]                              # Scores for ALL 3 labels
