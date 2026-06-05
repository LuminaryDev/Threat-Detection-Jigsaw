---
language:
- en
license: mit
tags:
- text-classification
- roberta
- threat-detection
- toxicity
- jigsaw
- multi-label
datasets:
- google/jigsaw_toxicity_pred
metrics:
- f1
model-index:
- name: Pommu/threat-detection-jigsaw
  results:
  - task:
      type: text-classification
    dataset:
      name: Jigsaw Toxic Comment Classification
      type: google/jigsaw_toxicity_pred
    metrics:
    - type: f1
      value: 0.79
      name: Weighted F1
---

# 🛡️ Context-Aware Threat & Compliance Detection in Conversational Text


**NLP Course Project — Individual Assignment**  
**Model**: Fine-tuned `roberta-base` | **Task**: 6-class Multi-label Classification  
**Dataset**: `google/jigsaw_toxicity_pred` (~150K samples)

---

## 🌐 Live Demo

| Resource | URL |
|---|---|
| 🖥️ **Interactive UI (HuggingFace Spaces)** | https://huggingface.co/spaces/Pommu/threat-detection-jigsaw |
| 🤖 **Fine-tuned Model** | https://huggingface.co/Pommu/threat-detection-jigsaw |
| 🔌 **Local API** | `http://localhost:8000/predict` (POST) |

---

## 📌 Problem Statement

Modern organizations face increasing risks from implicit threats and workplace harassment. Unlike simple slur-detectors, this project focuses on **intent-based detection**. It uses the Jigsaw Toxic Comment dataset to identify professional and personal risk across 6 binary categories:

| Label | Description | Risk Level |
|---|---|---|
| `Threatening` | Violent intent, intimidation, physical danger | 🔴 HIGH |
| `Hate Speech` | Identity-based attacks, protected groups | 🔴 HIGH |
| `Highly Severe` | Extreme toxicity, highly disruptive | 🔴 HIGH |
| `Toxic` | Rude, disrespectful, unprofessional | 🟡 MEDIUM |
| `Insult` | Personal attacks, non-violent harassment | 🟡 MEDIUM |
| `Profanity` | Obscene language, compliance violation | 🟡 LOW |

---

## 🏗️ Architecture: The 2-Layer Safety Net

```
[User Input: Text / Chat Transcript]
         │
         ▼
   [Gradio UI] ── HTTP POST /predict ──▶ [FastAPI Backend]
                                                │
                                   ┌────────────┴────────────┐
                                   │                         │
                             [Layer 1]                 [Layer 2]
                       Fine-tuned RoBERTa          Lexical Booster
                       (6 Jigsaw labels)       (keyword pattern match)
                                   │                         │
                                   └────────────┬────────────┘
                                                │
                                  {label, confidence, risk_level}
```

1. **Layer 1 — Weighted RoBERTa**: Multi-label model fine-tuned with **50x class weights** on the rare `threat` class (0.3% of data) to maximize recall.
2. **Layer 2 — Lexical Booster**: Backend safety layer that catches ominous linguistic patterns (e.g., "watch your back", "you'll regret") *unconditionally*, regardless of model score.

---

## 📊 Evaluation Results


> **Design Choice**: We prioritize *Recall* over *Precision* for threats. A false positive (flagging safe text) is far less costly than a false negative (missing a real threat).

---

## 🚀 API Usage

```bash
# Test implicit threat detection locally
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I will find you and you will regret every decision you have made."}'
```

**JSON Response:**
```json
{
  "prediction": "Threatening",
  "confidence": 0.85,
  "is_threat": true,
  "risk_level": "HIGH",
  "all_scores": [
    {"label": "Toxic", "confidence": 0.0009},
    {"label": "Threatening", "confidence": 0.85},
    ...
  ]
}
```

---

## 📂 Project Structure

```
NLP Course Project/
├── notebook/
│   └── train_3.ipynb         ← Colab training notebook (WeightedTrainer)
├── app/
│   ├── main.py               ← FastAPI backend + Lexical Booster
│   ├── model_loader.py       ← Singleton model loader (thread-safe)
│   ├── schemas.py            ← Pydantic request/response models
│   └── demo.py               ← Gradio UI (standalone OR API-backed)
├── spaces/
│   ├── app.py                ← HuggingFace Spaces entry point
│   └── requirements.txt
├── requirements.txt          ← API/demo dependencies
├── requirements_training.txt ← Training dependencies (Colab)
├── setup_local.bat           ← Windows one-click environment setup
└── README.md
```

---

## ⚙️ Local Setup (Windows)

```bat
# Option A: one-click setup
setup_local.bat

# Option B: manual
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your `.env` file:
```env
HF_TOKEN=your_token_here
MODEL_NAME=Pommu/threat-detection-jigsaw
```

---

## 🏃 Running Locally

```bash
# Terminal 1 — Start the FastAPI backend
uvicorn app.main:app --reload

# Terminal 2 — Launch the Gradio demo
python app/demo.py
```

Open `http://localhost:7860` to see the UI.

---

## ☁️ Training on Colab

1. Open `notebook/train_3.ipynb` in Google Colab
2. Enable GPU accelerator (Tesla T4)
3. Add `HF_Token` to Colab Secrets (🔑 icon)
4. Run all cells — model auto-pushes to HuggingFace Hub

---

## 🌐 Deploy to HuggingFace Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: Gradio
2. In Space Settings → Variables: `MODEL_NAME = Pommu/threat-detection-jigsaw`
3. Push `spaces/` contents to the Space repo

---

## 📚 Technologies Used

| Component | Technology |
|---|---|
| **Model** | `roberta-base` (HuggingFace Transformers) |
| **Training** | HuggingFace `Trainer` API + `WeightedTrainer` (custom) |
| **Dataset** | `google/jigsaw_toxicity_pred` (150K samples, 6 labels) |
| **Backend API** | FastAPI + Uvicorn |
| **UI** | Gradio |
| **Deployment** | HuggingFace Spaces |
| **Evaluation** | scikit-learn (Precision/Recall/F1/Confusion Matrix) |
