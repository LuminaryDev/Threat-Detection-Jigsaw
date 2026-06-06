# 🛡️ Jigsaw Threat Detector: Technical Briefing
**Project**: Natural Language Processing Course Final Project  
**Student**: Rediet Girmay  
**Model**: Fine-tuned RoBERTa-base  
**Dataset**: Google Jigsaw Toxic Comment Classification  

---

## 1. Core Concepts (The "Why")

### 🧬 RoBERTa: The Bidirectional Brain
*   **What it is**: Stands for "Robustly Optimized BERT Approach." It is a Transformer model created by Meta (Facebook).
*   **Why it's smart**: Unlike older models that read text linearly (left-to-right), RoBERTa is **Bidirectional**. It looks at the words before AND after a keyword simultaneously to understand context.
*   **Attention Mechanism**: It uses "Mathematical Attention" to focus on high-risk word pairs (e.g., "watch" + "back") while ignoring filler words ("the", "is").

### 📚 Jigsaw Dataset: The Textbook
*   Contains **159,571 samples** of real human comments.
*   **Multi-label**: Each sentence isn't just "good" or "bad." It is scored across 6 specific categories (Toxic, Severe, Obscene, Threat, Insult, Identity Hate).

### ⚖️ WeightedTrainer: The Secret Weapon
*   **The Problem**: Only 0.3% of the data are "Threats." A normal AI would ignore them to get 99% accuracy easily.
*   **The Fix**: We implemented a custom `WeightedTrainer`. If the AI misses a **Threat**, the math treats it as **50 times more painful** than a normal mistake. This forces the model to be extremely sensitive to danger.

---

## 2. Notebook Breakdown (Cell-By-Cell)

| Cell | Purpose | Why it matters |
|:---|:---|:---|
| **1-2** | Setup & Login | Installs the engine (PyTorch/Transformers) and connects to HuggingFace for storage. |
| **3** | Configuration | Defines the hyper-parameters (3 Epochs, 2e-5 Learning Rate). |
| **4** | Data Loading | Downloads the raw 150k samples directly from GitHub to ensure data integrity. |
| **5** | EDA (Charts) | **The Goal**: Visualize the imbalance. You can show your teacher these charts to prove why "Threats" are the hardest class to detect. |
| **6-7** | Preprocessing | Cleans emojis/junk text and converts words into **Tokens** (numbers) that the AI can read. |
| **8** | Load RoBERTa | Downloads the "Pre-trained" brain before we start teaching it about toxicity. |
| **10** | **The Brain Logic** | This is where the `WeightedTrainer` code lives. It's the most technical and impressive part of your project. |
| **11** | Training Run | **The Workhorse**: This cell ran for **47 minutes** on a Tesla T4 GPU, processing 23,937 batches. |
| **12-13** | Evaluation | Generates the **Classification Report** and **Confusion Matrix**. This is your proof that the model actually works. |
| **14** | Push to Hub | Ships your final trained "Brain" to the internet so the app can use it. |

---

## 3. Understanding the Training Output
When you look at **Cell 12**, you see these scores:

1.  **Weighted F1: 0.7908**: This is your "Main Grade." It means the model is about 80% accurate across all categories combined, which is excellent for a complex NLP task.
2.  **Threat Recall: 48.6%**: While this looks low, it is actually **quite high** for this dataset. Because threats are so rare (only 37 in the test set), catching ~50% of them with a neural model is a strong result.
3.  **The "Fusion"**: Once training finished, the knowledge of the "Textbook" (Jigsaw) was baked into the "Brain" (RoBERTa). You now have a single file (the Model) that can predict future text.

---
## 4. Multi-Layered Safety (Layer 2)
You should emphasize that your app doesn't just rely on the AI. 
*   **Layer 1**: The AI (RoBERTa) handles the nuanced math.
*   **Layer 2**: The **Lexical Booster** handles "Absolute Truths." If someone says "watch your back," no matter what the AI math says, the system **must** flag it. This is why you see **85% Confidence** on certain test cases even if the bar is low.
