"""
app/demo.py — Gradio UI (Standalone OR API-backed)
=====================================================
Developer-grade dark UI for the Threat & Compliance Detection system.

TWO MODES:
  1. STANDALONE (default): loads HuggingFace model directly in this process.
  2. API MODE  (USE_API_MODE=true): calls local FastAPI server (/predict).

HOW TO RUN:
  Standalone mode:  python app/demo.py
  API-backed mode:  USE_API_MODE=true python app/demo.py
"""

import os
import logging
import gradio as gr
import requests
from transformers import pipeline as hf_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
USE_API_MODE  = os.getenv("USE_API_MODE",  "false").lower() == "true"
API_URL       = os.getenv("API_URL",       "http://localhost:8000/predict")
HF_MODEL_NAME = os.getenv("MODEL_NAME",   "Pommu/threat-detection-jigsaw")

LABEL_MAP = {
    "toxic":         "Toxic",
    "severe_toxic":  "Highly Severe",
    "obscene":       "Profanity",
    "threat":        "Threatening",
    "insult":        "Insult",
    "identity_hate": "Hate Speech",
}

RISK_ICON = {
    "Safe":          ("🟢", "SAFE",     "LOW",    "#22c55e"),
    "Profanity":     ("🟡", "PROFANITY","LOW",    "#eab308"),
    "Toxic":         ("🟡", "TOXIC",    "MEDIUM", "#f97316"),
    "Insult":        ("🟠", "INSULT",   "MEDIUM", "#f97316"),
    "Hate Speech":   ("🔴", "HATE",     "HIGH",   "#ef4444"),
    "Highly Severe": ("🔴", "SEVERE",   "HIGH",   "#dc2626"),
    "Threatening":   ("🚨", "THREAT",   "HIGH",   "#dc2626"),
}

# ── Comprehensive Lexical Booster (UNCONDITIONAL on match) ─────────────────────
BOOSTER_PHRASES = [
    # Explicit locating
    "find you", "track you", "hunt you", "come for you", "know where you live",
    "know where you are", "i know where you",
    # Consequence threats
    "regret", "you will pay", "make you pay", "pay for this", "you'll regret",
    "won't forget this", "will not forget", "this isn't over",
    # Violence
    "mess you up", "beat you", "hurt you", "destroy you", "ruin you",
    "end you", "silence you", "take you out", "take care of you",
    "i'll get you", "i will get you", "gonna get you", "going to get you",
    # Financial/coercion
    "serious consequences", "face consequences", "pay the price",
    "watch your back", "watch yourself",
    # Warnings
    "last warning", "final warning", "don't push me", "pushing my limit",
]

# ── Model (lazy-loaded singleton) ──────────────────────────────────────────────
_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        logger.info(f"Loading model: {HF_MODEL_NAME}")
        _pipe = hf_pipeline(
            "text-classification",
            model=HF_MODEL_NAME,
            tokenizer="roberta-base",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("Model ready ✅")
    return _pipe


# ── Core prediction ────────────────────────────────────────────────────────────
def analyze(text: str):
    """Returns (markdown_result, status_dot_html)"""
    if not text or not text.strip():
        return "```\n⚠  Please enter some text to analyze.\n```", ""

    try:
        if USE_API_MODE:
            resp = requests.post(API_URL, json={"text": text}, timeout=30)
            resp.raise_for_status()
            data       = resp.json()
            prediction = data["prediction"]
            confidence = data["confidence"]
            all_scores = {s["label"]: s["confidence"] for s in data["all_scores"]}
            found_phrases = []   # API handles this internally

        else:
            # ── Lexical Booster (fires unconditionally) ──────────────────────
            text_lower    = text.lower()
            found_phrases = [p for p in BOOSTER_PHRASES if p in text_lower]

            raw         = _get_pipe()(text)[0]
            scores_dict = {r["label"]: r["score"] for r in raw}
            threat_score = scores_dict.get("threat", 0)
            hate_score   = scores_dict.get("identity_hate", 0)

            # Booster fires unconditionally on keyword match
            if found_phrases:
                prediction = "Threatening"
                confidence = max(threat_score, 0.85)
            elif threat_score > 0.02:
                prediction = "Threatening"
                confidence = threat_score
            elif hate_score > 0.05:
                prediction = "Hate Speech"
                confidence = hate_score
            else:
                raw_sorted = sorted(raw, key=lambda x: x["score"], reverse=True)
                top        = raw_sorted[0]
                prediction = LABEL_MAP.get(top["label"], top["label"])
                confidence = top["score"]

            all_scores = {LABEL_MAP.get(r["label"], r["label"]): r["score"] for r in raw}

        icon, badge, risk_level, color = RISK_ICON.get(
            prediction, ("⚪", "UNKNOWN", "UNKNOWN", "#6b7280")
        )

        action_text = (
            "Escalate to Security Team" if risk_level == "HIGH"
            else "Flag for HR Review" if risk_level == "MEDIUM"
            else "No action required"
        )

        action_icon = "🚨" if risk_level == "HIGH" else "⚠️" if risk_level == "MEDIUM" else "✅"

        # Detection method indicator
        if found_phrases:
            method_note = f"\n> 🔍 **Lexical Booster triggered** — matched phrase: `{found_phrases[0]}`"
        elif USE_API_MODE:
            method_note = "\n> 🔌 **API Mode** — result from FastAPI backend"
        else:
            method_note = "\n> 🧠 **Neural Model** — result from fine-tuned RoBERTa"

        # ── Build output markdown ──────────────────────────────────────────────
        md = f"""
<div style="background:#111827;border-left:4px solid {color};padding:16px 20px;border-radius:8px;margin-bottom:12px;font-family:monospace;">
  <div style="font-size:22px;font-weight:700;color:{color};">{icon} {badge}</div>
  <div style="color:#9ca3af;font-size:13px;margin-top:4px;">Risk Assessment Result</div>
</div>

| Field | Value |
|:---|:---|
| **Verdict** | `{prediction}` |
| **Risk Level** | `{risk_level}` |
| **Confidence** | `{confidence:.1%}` |
| **Recommended Action** | {action_icon} {action_text} |
| **Detection Layer** | {'Layer 2 — Lexical Booster' if found_phrases else 'Layer 1 — Neural Model'} |
{method_note}

---
### 📊 All Category Scores
"""
        # Sort: Threatening first, then by score
        priority_order = ["Threatening", "Hate Speech", "Highly Severe", "Toxic", "Insult", "Profanity"]
        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: (priority_order.index(x[0]) if x[0] in priority_order else 99, -x[1])
        )

        for label, score in sorted_scores:
            _, _, lvl, bar_color = RISK_ICON.get(label, ("", "", "LOW", "#4b5563"))
            filled = round(score * 20)
            bar    = "█" * filled + "░" * (20 - filled)
            highlight = " ◀ **FLAGGED**" if label == prediction and score > 0.01 else ""
            md += f"\n`{label:<14}` `{bar}` `{score:.1%}`{highlight}"

        return md

    except Exception as e:
        logger.error(f"Error: {e}")
        return f"```\n❌ Error: {e}\n```"


# ── Examples (high-scoring, interesting cases) ─────────────────────────────────
EXAMPLES = [
    ["I will find you and make you regret everything."],
    ["I know where you live. Watch your back."],
    ["Transfer the funds or face serious consequences."],
    ["You are an absolute disgrace, worthless and incompetent."],
    ["We need to push back on this deadline — please respond ASAP."],
    ["People like you shouldn't be allowed in this country."],
]

# ── Custom CSS (developer dark mode) ──────────────────────────────────────────
CSS = """
/* ── Root & Background ─────────────────────────── */
body, .gradio-container {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
}

/* ── Header ────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
}
.app-header h1 {
    font-size: 24px;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header .subtitle {
    font-size: 13px;
    color: #8b949e;
    margin-bottom: 14px;
}
.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.badge {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    color: #8b949e;
}
.badge.green { border-color: #238636; color: #3fb950; }
.badge.red   { border-color: #da3633; color: #f85149; }
.badge.blue  { border-color: #1f6feb; color: #58a6ff; }

/* ── Textbox ────────────────────────────────────── */
textarea, .gr-textbox textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 12px !important;
}
textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

/* ── Buttons ────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px) !important;
}
button.secondary {
    background: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Output Markdown ─────────────────────────────── */
.gr-markdown, .output-markdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 16px !important;
    color: #e6edf3 !important;
}
.gr-markdown table {
    border-collapse: collapse;
    width: 100%;
}
.gr-markdown td, .gr-markdown th {
    padding: 6px 12px;
    border: 1px solid #30363d;
    font-family: monospace;
    font-size: 13px;
}
.gr-markdown th { background: #21262d; color: #8b949e; }
.gr-markdown code {
    background: #21262d;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
    color: #79c0ff;
}
.gr-markdown blockquote {
    border-left: 3px solid #388bfd;
    padding: 8px 12px;
    background: #161b22;
    margin: 8px 0;
    color: #8b949e;
    font-size: 12px;
}

/* ── Examples ───────────────────────────────────── */
.gr-examples {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Label (textbox title) ───────────────────────── */
label span {
    color: #8b949e !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
"""

# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="ThreatScan — Enterprise Compliance Detection",
    css=CSS,
) as demo:

    gr.HTML("""
    <div class="app-header">
      <h1>🛡️ ThreatScan — Threat &amp; Compliance Detection</h1>
      <div class="subtitle">Context-aware analysis of conversational text for threats, harassment, and compliance violations.</div>
      <div class="badge-row">
        <span class="badge blue">🤖 roberta-base</span>
        <span class="badge green">📦 Jigsaw Toxic Comments Dataset</span>
        <span class="badge green">6-class Multi-label</span>
        <span class="badge red">2-Layer Safety Net</span>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            txt_in = gr.Textbox(
                label="INPUT TEXT",
                placeholder="Paste a message, call transcript line, or any conversational text…",
                lines=6,
                show_label=True,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(txt_in, value="🗑  Clear", variant="secondary")
                analyze_btn = gr.Button("🔍  Analyze", variant="primary", scale=3)

            gr.Examples(
                examples=EXAMPLES,
                inputs=txt_in,
                label="▶ Quick Test Cases",
            )

        with gr.Column(scale=7):
            output = gr.Markdown(
                value="""
> **System Ready.** Paste a text on the left and click **Analyze** to run the threat classifier.

```
Model  : Pommu/threat-detection-jigsaw
Task   : Multi-label classification (6 categories)
Data   : google/jigsaw_toxicity_pred (~150K samples)
Method : Fine-tuned RoBERTa + Lexical Booster
```
""",
            )

    gr.HTML("""
    <div style="margin-top:16px;padding:12px 20px;background:#161b22;border:1px solid #30363d;
         border-radius:8px;font-size:12px;color:#484f58;text-align:center;font-family:monospace;">
      ThreatScan v1.0 · Academic Prototype · Predictions are probabilistic ·
      <a href="https://huggingface.co/Pommu/threat-detection-jigsaw" style="color:#58a6ff;">
        Model on HuggingFace ↗
      </a>
    </div>
    """)

    analyze_btn.click(fn=analyze, inputs=txt_in, outputs=output)
    txt_in.submit(fn=analyze, inputs=txt_in, outputs=output)


if __name__ == "__main__":
    print(f"{'='*55}")
    print(f"  ThreatScan UI  |  Mode: {'API' if USE_API_MODE else 'Standalone'}")
    print(f"  Model : {HF_MODEL_NAME}")
    print(f"  Open  : http://localhost:7860")
    print(f"{'='*55}")
    demo.launch(server_port=7860, share=False, show_error=True)
