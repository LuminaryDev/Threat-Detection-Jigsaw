# NLP Course Project: Technical Report
**Institution**: Addis Ababa University  
**Program**: MSc in Artificial Intelligence  
**Course**: Natural Language Processing (NLP)  
**Name**: Rediet Girmay  
**ID**: GSE-0945/17  
**Assignment**: Context-Aware Threat & Compliance Detection  

---

## 1. Project Introduction
This project addresses the challenge of identifying threatening and non-compliant speech in conversational text. While traditional systems rely on static "blacklists" of keywords, this implementation uses a **Transformer-based Deep Learning** approach to understand linguistic context and intent. The goal is to provide a robust auditing tool for enterprise communications, capable of flagging threats, harassment, and identity-based hate speech.

## 2. Technical Methodology

### 2.1 Model Architecture
The core of the system is built on **RoBERTa (Robustly Optimized BERT Approach)**. RoBERTa was selected for its superior understanding of subtle linguistic nuances and its robust performance on downstream classification tasks.

### 2.2 Dataset: Jigsaw Toxicity Pred
We utilized the **Google Jigsaw Toxic Comment Classification** dataset, comprising approximately 150,000 labeled samples across 6 distinct categories:
1.  **Toxic**: Rude or disrespectful language.
2.  **Severe Toxic**: Highly disruptive or damaging toxicity.
3.  **Obscene**: Profanity and obscene content.
4.  **Threat**: Violences, intimidation, or physical harm.
5.  **Insult**: Personal attacks and harassment.
6.  **Identity Hate**: Attacks based on protected group identification.

### 2.3 Optimization Strategy: WeightedTrainer
A significant challenge in threat detection is **Class Imbalance**. In the Jigsaw dataset, "Threats" represent only ~0.3% of the samples. To avoid a model that simply ignores threats to achieve high accuracy, I implemented a custom **WeightedTrainer**. This trainer applies a **50x weight penalty** to missed threats during backpropagation, forcing the model to prioritize **Recall** (finding the threats) over simple precision.

## 3. Evaluation & Performance Metrics

The model underwent 3 epochs of fine-tuning on a Tesla T4 GPU. The final evaluation on the test set yielded the following per-class F1-scores:

| Category | Precision | Recall | F1-Score |
|:---|:---|:---|:---|
| **Toxic** | 0.8621 | 0.8155 | 0.8382 |
| **Severe Toxic** | 0.4068 | 0.6486 | 0.5000 |
| **Obscene** | 0.8441 | 0.8098 | 0.8266 |
| **Threat** | 0.4500 | 0.4865 | **0.4675** |
| **Insult** | 0.7748 | 0.7611 | 0.7679 |
| **Identity Hate** | 0.5723 | 0.6463 | 0.6070 |

### 📊 Global Averages
*   **Weighted F1 Score**: **0.7908** (Primary performance metric)
*   **Macro F1 Score**: **0.6679** (Conservative average across imbalanced classes)

## 4. Multi-Layered Safety Architecture
Recognizing that neural models can occasionally miss implicit threats (e.g., "I know where you live"), I implemented a **Lexical Booster** in the inference pipeline. This layer scans for 30+ ominous linguistic patterns and escalates the risk level to **HIGH** even if the neural logits are low. This ensures an enterprise-grade "Safety-First" configuration.

## 5. Deployment & Links
The implementation is fully open-sourced and deployed for interactive testing:

*   **GitHub Repository**: [https://github.com/LuminaryDev/Threat-Detection-Jigsaw](https://github.com/LuminaryDev/Threat-Detection-Jigsaw)
*   **HuggingFace Spaces (Demo)**: [https://huggingface.co/spaces/Pommu/threat-detection-jigsaw](https://huggingface.co/spaces/Pommu/threat-detection-jigsaw)
*   **Model Hub**: [https://huggingface.co/Pommu/threat-detection-jigsaw](https://huggingface.co/Pommu/threat-detection-jigsaw)

---


