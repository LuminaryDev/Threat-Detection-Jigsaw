# 📘 Master Project Guide: Jigsaw Threat Detection
**Student**: Rediet Girmay (GSE-0945/17)  
**Course**: NLP (MSc in AI)  
**Institution**: Addis Ababa University  

This guide is your single source of truth for understanding, explaining, and presenting your project.

---

## 1. File Structure (The Anatomy)

### 📂 Root Directory
*   **[.env](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/.env)**: Stores your HuggingFace Token and Model Name. Keeps your secrets safe.
*   **[SUBMISSION_REPORT.md](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/SUBMISSION_REPORT.md)**: Your formal academic report for your professor.
*   **[README.md](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/README.md)**: The main project page that shows up on GitHub/HuggingFace.

### 📂 `notebook/` (The Lab)
*   **[train.ipynb](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/notebook/train.ipynb)**: Your core training pipeline. This is where the AI "learned" how to detect threats using the Jigsaw dataset and RoBERTa.

### 📂 `app/` (The Application Backend)
*   **[main.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/app/main.py)**: The FastAPI server. It manages the detection requests and runs the **Lexical Booster** (Layer 2).
*   **[model_loader.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/app/model_loader.py)**: A helper script that downloads the "Brain" (Model) from HuggingFace and gets it ready for predictions.
*   **[demo.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/app/demo.py)**: The local Gradio UI code. This is what you see when you run the app on your computer.

### 📂 `spaces/` (The Deployed Version)
*   **[app.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/spaces/app.py)**: A self-contained version of your app specifically built for HuggingFace Spaces. It includes the AI brain + the Lexical Booster.

---

## 2. Dataset vs. Model (The Teacher vs. The Student)

*   **Jigsaw Dataset (The Textbook)**: A collection of 150,000+ real internet comments. It has labels created by humans identifying if a comment is toxic, a threat, etc. This is the **Input** to the training process.
*   **RoBERTa (The Brain)**: A pre-trained neural network that already understands English grammar and language perfectly.
*   **The Model (The Expert)**: After we run the training notebook, RoBERTa "absorbs" the Jigsaw data. The final result is a **Model**—a trained expert that can now look at a NEW sentence and predict its toxicity.

---

## 3. The 3-Way Data Split (Train/Val/Test)

1.  **Training Set**: The AI reads this to learn patterns.
2.  **Validation Set (Practice Quiz)**: Used *during* training to see if the AI is memorizing or actually learning. We use this to stop training early if it plateaus (**Early Stopping**).
3.  **Test Set (Final Exam)**: Used at the very end. The AI has never seen this data. It provides the final "True" score of 0.79 Weighted F1.

---

## 4. Key Concepts for Q&A

### 🧮 Understanding the Code: The "compute_metrics" logic
Your code uses a mathematical function called **Sigmoid**:
*   **Why?**: In a standard AI (like dog vs. cat), you use *Argmax* (pick only one).
*   **Multi-label**: In our project, a comment can be **Toxic** AND an **Insult** at the same time. **Sigmoid** calculates a 0%–100% score for *each* category independently.
*   **Threshold (0.5)**: We told the AI: "If you are more than 50% sure about a category, check the box for it."

---

## 4. The "Three Pillars" of Evaluation (Metrics)

1.  **Precision**: "Of all the times the AI said 'Threat', how many were *actually* threats?" (Goal: No false alarms).
2.  **Recall**: "Of all the real threats in the data, how many did the AI find?" (Goal: Don't miss anyone).
3.  **F1-Score**: The harmonic mean of the two. It's the "True Grade" because you can't cheat by just guessing "Everything is Toxic."

---

## 5. Deployment Fine-Tuning (The 2% Logic)
In our application code ([app/main.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/app/main.py)), we did something clever:
*   **The Logic**: Neural models are sometimes shy. Even if the AI only gives a **2% score (0.02)** to a threat, our app is programmed to show a **Warning**.
*   **Why?**: For security, we prefer a "False Alarm" (over-sensitivity) rather than a "Missed Threat."
*   **The Fusion**: The app combines the **AI score** (even low ones) with the **Lexical Booster** (high-risk words) to give the final verdict.

---

## 5. Deployment Logic (How it works on the Web)

1.  **Step 1**: You train the model in Colab ([notebook/train.ipynb](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/notebook/train.ipynb)).
2.  **Step 2**: You `push` the model to the **HuggingFace Hub** (`Pommu/threat-detection-jigsaw`).
3.  **Step 3**: Your HuggingFace Space ([app.py](file:///C:/Users/pc/OneDrive/Desktop/Pommy/NLP/NLP%20Course%20Project/spaces/app.py)) `pulls` that model and runs it in the cloud.
4.  **The Result**: Anyone with your link can test the final neural + lexical system in real-time.

---

# 🎓 Final Presentation Advice
If the professor asks: *"Why is your threat score lower than your toxic score?"*
**Answer**: *"Threats are rare—representing less than 1% of the training data. This 'Class Imbalance' makes them harder to learn. However, we used a **WeightedTrainer with 50x penalty** to maximize sensitivity, and implemented a **Hybrid Lexical Booster** in our application to ensure that even rare threats are captured with high confidence."*
