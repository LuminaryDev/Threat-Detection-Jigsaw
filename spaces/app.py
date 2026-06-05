# ============================================================
# spaces/app.py — HuggingFace Spaces Entry Point
# ============================================================
# For deployment on HuggingFace Spaces (Gradio SDK):
#
# 1. Create a new Space at huggingface.co/new-space
#    - SDK: Gradio
#    - Visibility: Public
#
# 2. In the Space settings → Variables & Secrets:
#    - Add Secret:  HF_TOKEN = your_write_token
#    - Add Variable: MODEL_NAME = your_hf_username/threat-detection-roberta
#
# 3. Push these files to the Space repo:
#    - spaces/app.py  (rename to app.py at root)
#    - spaces/requirements.txt
#
# The Space will auto-build and give you a public URL.
# ============================================================

import sys, os
# Allow importing from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.demo import demo   # Reuse the Gradio demo we already built

if __name__ == "__main__":
    # share=False because HF Spaces handles public URL automatically
    demo.launch()
