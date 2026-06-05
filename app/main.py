"""
app/main.py — FastAPI Backend: Threat & Compliance Detection API
================================================================
CONCEPT: What is FastAPI?
FastAPI is a modern Python web framework for building REST APIs.
"REST API" = a standard where clients send HTTP requests (GET/POST)
and receive structured JSON responses.

Key parts of this file:
  1. lifespan()     — startup hook to warm up the model
  2. CORS middleware — allows browser requests from other domains (Gradio UI)
  3. /predict       — the main endpoint that runs inference
  4. /health        — simple check to confirm the server is alive

HOW TO RUN LOCALLY:
  uvicorn app.main:app --reload --port 8000

Then test with:
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"text": "I will make you pay for this."}'
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import PredictRequest, PredictResponse, ScoreDetail
from app.model_loader import get_pipeline, LABEL_MAP, RISK_MAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────
# Code inside `lifespan` runs ONCE when the server starts (before `yield`)
# and again when it shuts down (after `yield`).
# Warming the model here means the FIRST user request won't be slow.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting — warming up model...")
    get_pipeline()          # Load & cache the model now
    yield                   # ← Server runs here, handling requests
    logger.info("Server shutting down.")


# ── App Instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Threat & Compliance Detection API",
    description=(
        "Detects threatening speech, harassment, and compliance violations "
        "in conversational text using a fine-tuned RoBERTa model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows your Gradio UI (running on a
# different port or domain) to call this API without browser security errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production, restrict to your Spaces URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Root / Health Check")
async def root():
    return {"status": "online", "service": "Threat Detection API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, summary="Analyze text for threats")
async def predict(request: PredictRequest):
    """
    Accepts a JSON body: {"text": "..."} 
    Returns the predicted threat category + confidence scores for all 3 labels.
    """
    try:
        logger.info(f"Analyzing: '{request.text[:60]}...'")

        pipe = get_pipeline()

        # pipe() returns: [[{"label": "LABEL_0", "score": 0.91}, ...]]
        # top_k=None means we get scores for ALL labels (set in model_loader)
        raw_results = pipe(request.text)[0]

        # ── MULTI-LABEL CALIBRATION ─────────────────────────────────────────
        # In a multi-label model (Jigsaw), multiple things can be true.
        # We look for any category above 30%, prioritizing 'Threatening'.
        
        scores_dict = {r["label"]: r["score"] for r in raw_results}
        
        # High priority labels
        threat_score = scores_dict.get("threat", 0)
        toxic_score  = scores_dict.get("toxic", 0)
        hate_score   = scores_dict.get("identity_hate", 0)
        
        # ── LEXICAL BOOSTER ────────────────────────────────────────────────
        # Safety layer: catches ominous patterns the model under-scores.
        # These fire UNCONDITIONALLY — model score is NOT required.
        booster_phrases = [
            "find you", "regret", "make you pay", "consequences",
            "watch your back", "know where you", "repay", "repayment",
            "i will get you", "come for you", "pay for this",
            "you will pay", "won't forget", "you'll regret",
            "hunt you", "track you", "end you", "silence you",
            "mess you up", "mark my words", "bad things happen"
        ]
        found_phrases = [p for p in booster_phrases if p in request.text.lower()]

        # Determine top prediction
        # NOTE: Threshold set to 2% (0.02) to balance safety vs false positives.
        if found_phrases:
            top_label = "Threatening"
            top_score = max(threat_score, 0.85)   # Always report high confidence
            logger.info(f"🚨 LEXICAL BOOSTER TRIGGERED: {found_phrases[0]}")
        elif threat_score > 0.02:
            top_label = "Threatening"
            top_score = threat_score
        elif hate_score > 0.05:
            top_label = "Hate Speech"
            top_score = hate_score
        else:
            # Fall back to highest score overall
            sorted_results = sorted(raw_results, key=lambda x: x["score"], reverse=True)
            top_raw   = sorted_results[0]["label"]
            top_score = sorted_results[0]["score"]
            top_label = LABEL_MAP.get(top_raw, top_raw)

        return PredictResponse(
            input_text   = request.text,
            prediction   = top_label,
            confidence   = round(top_score, 4),
            is_threat    = bool(found_phrases) or any(s["score"] > 0.25 for s in raw_results if s["label"] != "obscene"),
            risk_level   = RISK_MAP.get(top_label, "LOW"),
            all_scores   = [
                ScoreDetail(
                    label      = LABEL_MAP.get(r["label"], r["label"]),
                    confidence = round(r["score"], 4),
                )
                for r in raw_results # Show all 6 for better compliance visibility
            ],
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
