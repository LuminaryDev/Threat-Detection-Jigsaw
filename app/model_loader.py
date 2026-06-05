"""
app/model_loader.py — Singleton Model Loader
=============================================
CONCEPT: The Singleton Pattern
A "singleton" means only ONE instance of an object ever exists.
Here, we load the ML model ONCE when the server starts, then
reuse that same loaded model for every single prediction request.

WHY THIS MATTERS:
- Loading a RoBERTa model takes ~8 seconds and ~500MB RAM
- If we loaded it fresh per-request, every user would wait 8+ seconds
- With a singleton: first request = 8 seconds, ALL subsequent = ~100ms

CONCEPT: Thread Safety (the _lock)
If two requests arrive at the exact same millisecond, two threads
might both try to load the model simultaneously — wasting memory.
The threading.Lock() prevents this: only ONE thread can enter the
"loading" block at a time. The double-check (`if _pipeline is None`
inside the lock) is a classic "Double-Checked Locking" pattern.
"""
import os
import threading
import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

# ⚠️  UPDATE THIS before deployment:
# Either set the MODEL_NAME environment variable, or change the default value.
# Format: "your_hf_username/your_model_repo_name"
HF_MODEL_NAME = os.getenv("MODEL_NAME", "Pommu/threat-detection-jigsaw")

# Jigsaw Label Mapping (6 labels)
LABEL_MAP = {
    "toxic":         "Toxic",
    "severe_toxic":  "Highly Severe",
    "obscene":       "Profanity",
    "threat":        "Threatening",
    "insult":        "Insult/Harassment",
    "identity_hate": "Hate Speech",
}

# Unified Risk Levels
RISK_MAP = {
    "Threatening":     "HIGH",
    "Highly Severe":   "HIGH",
    "Hate Speech":     "HIGH",
    "Toxic":           "MEDIUM",
    "Insult/Harassment": "MEDIUM",
    "Profanity":       "LOW",
    "Safe":            "LOW",
}

# Module-level state (private — do NOT import these directly)
_pipeline = None
_lock = threading.Lock()


def get_pipeline():
    """
    Returns the cached HuggingFace pipeline.
    Thread-safe. Loads from HF Hub on first call, cached for all subsequent calls.
    """
    global _pipeline

    if _pipeline is None:                   # Fast path: skip lock if already loaded
        with _lock:
            if _pipeline is None:           # Double-check inside lock
                logger.info(f"Loading model from Hub: {HF_MODEL_NAME}")
                HF_TOKENIZER_NAME = "roberta-base"
                _pipeline = pipeline(
                    task="text-classification",
                    model=HF_MODEL_NAME,
                    tokenizer=HF_TOKENIZER_NAME,
                    top_k=None,             # Return scores for ALL labels, not just top-1
                    device=-1,              # -1 = CPU. Change to 0 if you have a GPU
                    truncation=True,
                    max_length=512,
                )
                logger.info("Model loaded and cached ✅")

    return _pipeline
