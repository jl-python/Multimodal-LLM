# Week 8 — Multimodal LLM Streamlit App (Track C)

## Overview
This Streamlit app integrates the **Week 8 Multimodal LLM Tracks A & B** into a unified interface.  
It demonstrates **speech**, **text**, and **visualization** routing through a lightweight multimodal pipeline.

### Features
- 🎤 **Speech → LLM → Speech**: Record or upload a `.wav` file, transcribe with Whisper, answer with TinyLlama, and hear the response with gTTS.  
- 💬 **Text Query → Visualization**: Type natural-language queries and view validated Matplotlib charts.  
- 🧭 **Router Logic**: Automatically decides whether to use the LLM, Speech, or Visualization tool based on input type or keywords.  
- 📊 **Reproducibility Logs**: Automatically generates `week8_run_config.json` and `env_week8.json` for grading and reproducibility.

---

## Installation

### Clone or download the repo
```bash
git clone https://github.com/jl-python/Multimodal-LLM.git
cd "Track C - Multimodal LLM Streamlit"
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit
```bash
streamlit run app.py
```



