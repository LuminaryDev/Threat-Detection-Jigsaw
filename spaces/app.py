"""
app.py — HuggingFace Spaces Entry Point (Self-contained)
This file must be at the ROOT of the HuggingFace Space repository.
"""
import os
import logging
import gradio as gr
from transformers import pipeline as hf_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_MODEL_NAME = os.getenv("MODEL_NAME", "Pommu/threat-detection-jigsaw")

LABEL_MAP = {
    "toxic": "Toxic", "severe_toxic": "Highly Severe", "obscene": "Profanity",
    "threat": "Threatening", "insult": "Insult", "identity_hate": "Hate Speech",
}
RISK_ICON = {
    "Safe":          ("🟢", "SAFE",     "LOW"),
    "Profanity":     ("🟡", "PROFANITY","LOW"),
    "Toxic":         ("🟡", "TOXIC",    "MEDIUM"),
    "Insult":        ("🟠", "INSULT",   "MEDIUM"),
    "Hate Speech":   ("🔴", "HATE",     "HIGH"),
    "Highly Severe": ("🔴", "SEVERE",   "HIGH"),
    "Threatening":   ("🚨", "THREAT",   "HIGH"),
}
BOOSTER_PHRASES = [
    "find you", "regret", "make you pay", "consequences", "watch your back",
    "know where you", "repay", "i will get you", "come for you", "pay for this",
    "you will pay", "won't forget", "you'll regret", "hunt you", "track you",
    "end you", "silence you", "mess you up", "mark my words", "bad things happen",
    "serious consequences", "watch yourself", "last warning", "final warning",
]

_pipe = None

def _get_pipe():
    global _pipe
    if _pipe is None:
        logger.info(f"Loading model: {HF_MODEL_NAME}")
        _pipe = hf_pipeline(
            "text-classification", model=HF_MODEL_NAME,
            tokenizer="roberta-base", top_k=None, truncation=True, max_length=512,
        )
        logger.info("Model ready ✅")
    return _pipe


def analyze(text: str):
    if not text or not text.strip():
        return "⚠️ Please enter some text."
    try:
        text_lower = text.lower()
        found_phrases = [p for p in BOOSTER_PHRASES if p in text_lower]

        raw = _get_pipe()(text)[0]
        scores_dict = {r["label"]: r["score"] for r in raw}
        threat_score = scores_dict.get("threat", 0)
        hate_score = scores_dict.get("identity_hate", 0)

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
            top = raw_sorted[0]
            prediction = LABEL_MAP.get(top["label"], top["label"])
            confidence = top["score"]

        all_scores = {LABEL_MAP.get(r["label"], r["label"]): r["score"] for r in raw}
        icon, badge, risk = RISK_ICON.get(prediction, ("⚪", "UNKNOWN", "UNKNOWN"))
        action = "🚨 Escalate to Security" if risk == "HIGH" else "⚠️ Flag for HR Review" if risk == "MEDIUM" else "✅ No action required"
        layer = f"🔍 Lexical Booster triggered: `{found_phrases[0]}`" if found_phrases else "🧠 Neural Model (fine-tuned RoBERTa)"

        md = f"""## {icon} {badge}

| Field | Value |
|:---|:---|
| **Verdict** | `{prediction}` |
| **Risk Level** | `{risk}` |
| **Confidence** | `{confidence:.1%}` |
| **Action** | {action} |

> {layer}

---
### 📊 All Category Scores
"""
        priority = ["Threatening", "Hate Speech", "Highly Severe", "Toxic", "Insult", "Profanity"]
        for label, score in sorted(all_scores.items(), key=lambda x: (priority.index(x[0]) if x[0] in priority else 99, -x[1])):
            filled = round(score * 20)
            bar = "█" * filled + "░" * (20 - filled)
            flag = " ◀ **FLAGGED**" if label == prediction and score > 0.01 else ""
            md += f"\n`{label:<14}` `{bar}` `{score:.1%}`{flag}"
        return md
    except Exception as e:
        return f"❌ Error: {e}"


EXAMPLES = [
    ["I will find you and make you regret everything."],
    ["I know where you live. Watch your back."],
    ["Transfer the funds or face serious consequences."],
    ["You are an absolute disgrace, worthless and incompetent."],
    ["We need to push back on this deadline — please respond ASAP."],
    ["People like you shouldn't be allowed in this country."],
]

with gr.Blocks(title="🛡️ ThreatScan — Threat & Compliance Detection",
               theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate")) as demo:

    gr.Markdown("""
# 🛡️ ThreatScan — Enterprise Threat & Compliance Detection
**Context-aware analysis of conversational text for threats, harassment, and compliance violations.**

| 🤖 Model | 📦 Dataset | 🏷️ Labels | ⚙️ Architecture |
|---|---|---|---|
| `roberta-base` | Jigsaw Toxic Comments (150K) | 6-class multi-label | 2-Layer Safety Net |

---
""")

    with gr.Row():
        with gr.Column(scale=4):
            txt_in = gr.Textbox(label="📝 Input Text",
                                placeholder="Paste a message, call transcript, or email…",
                                lines=6)
            with gr.Row():
                gr.ClearButton(txt_in, value="🗑 Clear")
                btn = gr.Button("🔍 Analyze", variant="primary", scale=3)
            gr.Examples(examples=EXAMPLES, inputs=txt_in, label="▶ Quick Test Cases")

        with gr.Column(scale=6):
            output = gr.Markdown(value="> **System Ready.** Paste text on the left and click **Analyze**.")

    btn.click(fn=analyze, inputs=txt_in, outputs=output)
    txt_in.submit(fn=analyze, inputs=txt_in, outputs=output)

    gr.Markdown("---\n*ThreatScan v1.0 · Academic Prototype · Predictions are probabilistic.*")

if __name__ == "__main__":
    demo.launch()
